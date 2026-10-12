# S01：天数分页 GQA 注意力优化

本篇解释当前 B1 怎样在输入处理阶段复用分页 K/V 历史，减少多个查询位置重复读取同一批数据。MiniCPM5-2B 的 16 个查询头共用 2 个 KV 头，每个 KV 头对应 8 个查询头；这个头分组与一次处理多少查询位置是两个不同的维度。当前 kernel 可处理 2、8 或 16 个查询位置，密集长查询使用 Q16，短查询保留较窄 tile。下面沿页表寻址、在线 softmax 和调用入口解释实现，完整 Decode 源码见 [S02](S02-天数纯Decode直接序列索引与分页归约优化.md)，精度与正式吞吐结果见 [S03 第 6 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#6-精度与官方正式结果)，Mixed 分派、GEMM 与 CPU 反射优化见 S03。

## 1. 模型、运行环境与设计目标

模型使用 BF16，42 层，hidden size 2048、intermediate size 6144，查询头/KV 头/每头维度为 16/2/128，最大上下文 131072。KV 缓存每 token 为 `42 × 2 × 2 × 128 × 2 = 43008 bytes`，约 42 KiB；64 个并发请求各含 16K 输入和 1K 输出约需 44.6 GiB，尚未计入权重、激活和工作区。分页读取与服务中的缓存压力因此同时影响端到端时间。

当前平台为官方 OnlineLab BI-V150，显存 32 GiB，最终排名由组委会复现；软件为 Python 3.12.11、Torch 2.10.0 CoreX、CoreX 4.5.0、FlagTree 0.6.2a3+iluvatar3.6 和 Triton 3.6。服务使用镜像 site-packages 的 vLLM 0.24 vendor，安装包 commit 未确认；FlagGems 基底为 `1bbd3c585de61b9c7850dead56fb3b01ee81a546`，plugin 基底为 `47882ba2585dafdec92f06aff8ceb8af1de54630`。当前生产组合包含 Q2/Q8/Q16 Prefill、共同历史 64-key 读取、设备端 32/64-key Decode、Mixed q1 分派及 Prefill/Decode 的 runtime 数量参数，经原 TRITON_ATTN 枚举的受限实现子类进入运行链。Q16 是查询位置 tile，不是把模型的 16 个查询头改成另一种头配置。

实际模型诊断确认了 native Triton Attention 路径及其 mixed 窗口热点。当前实现从跨查询复用 K/V、完整历史与因果边界拆分、页表地址复用和数量参数编译策略四处减少工作。

## 2. 当前 Prefill、共享分页读取与在线 softmax

源文件为 [paged_gqa_prefill.py](../../FlagGems/src/flag_gems/runtime/backend/_iluvatar/fused/paged_gqa_prefill.py)。下面连续收录该生产文件，每段最多 40 行。输入和输出保持 BF16，在线 softmax 的最大值、质量及累积值使用 FP32，最终写回调用方 output。兼容判断及未知路径 fallback 由 S03 的 plugin 负责。

### 2.1 分页地址与页表复用

```python
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Paged BF16 GQA: reuse each KV tile across eight queries and eight heads."""

import triton
import triton.language as tl


@triton.jit
def _paged_gqa_tile_update(
    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
    begin, seq_len, scale, m, l, acc,
    BLOCK_SIZE: tl.constexpr, BLOCKS: tl.constexpr, TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
    KS2: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr,
    VS2: tl.constexpr, CAUSAL_BOUNDARY: tl.constexpr,
    KV_TILE: tl.constexpr = 32, DEFER_V: tl.constexpr = False,
):
    lanes = tl.arange(0, KV_TILE)
    j = begin + lanes
    logical = j // BLOCK_SIZE
    page_valid = logical < TABLE_COLS
    if CAUSAL_BOUNDARY:
        page_valid = page_valid & (j < seq_len)
    # Read page IDs at page granularity for aligned 32/64-key tiles.
    # Load page IDs once at page granularity, not once for every token lane.
    if BLOCK_SIZE == 16:
        first_page = begin // 16
        p0 = tl.load(Table + seq * TABLE_STRIDE + first_page,
                     (first_page < TABLE_COLS) & (begin < seq_len), 0)
        p1 = tl.load(Table + seq * TABLE_STRIDE + first_page + 1,
                     (first_page + 1 < TABLE_COLS) & (begin + 16 < seq_len), 0)
        if KV_TILE == 64:
            p2 = tl.load(Table + seq * TABLE_STRIDE + first_page + 2,
                         (first_page + 2 < TABLE_COLS) & (begin + 32 < seq_len), 0)
            p3 = tl.load(Table + seq * TABLE_STRIDE + first_page + 3,
                         (first_page + 3 < TABLE_COLS) & (begin + 48 < seq_len), 0)
            physical = tl.where(lanes < 32, tl.where(lanes < 16, p0, p1),
                                tl.where(lanes < 48, p2, p3)).to(tl.int64)
```

这段 helper 接收已经载入的 Q、分页 K/V 指针及上一轮 `m/l/acc`，首先把逻辑 key 位置 `j=begin+lanes` 转成逻辑页号。`KV_TILE` 为 32 或 64，而分页大小为 16 时，每 16 个 lane 共用一项页表，因此窄 tile 只需取两个标量页号，宽 tile 取四个。比如 `begin=64、KV_TILE=64` 覆盖逻辑页 4 到 7，四个页号分别广播给对应的 16 个位置，分页映射仍完整保留。页号加载检查页表容量和该页起点是否在真实序列内，后续才能用页内偏移访问缓存。这样减少的是同页 lane 重复取得页号的工作，K/V 数据仍根据物理页逐位置读取，具体收益需要由完整组合的测量判断。

```python
        else:
            physical = tl.where(lanes < 16, p0, p1).to(tl.int64)
    elif BLOCK_SIZE == 32 and KV_TILE == 64:
        first_page = begin // 32
        p0 = tl.load(Table + seq * TABLE_STRIDE + first_page,
                     (first_page < TABLE_COLS) & (begin < seq_len), 0)
        p1 = tl.load(Table + seq * TABLE_STRIDE + first_page + 1,
                     (first_page + 1 < TABLE_COLS) & (begin + 32 < seq_len), 0)
        physical = tl.where(lanes < 32, p0, p1).to(tl.int64)
    elif BLOCK_SIZE % KV_TILE == 0:
        first_page = begin // BLOCK_SIZE
        physical = tl.load(Table + seq * TABLE_STRIDE + first_page,
                           (first_page < TABLE_COLS) & (begin < seq_len), 0).to(tl.int64)
        physical = physical + tl.zeros((KV_TILE,), tl.int64)
    else:
        physical = tl.load(Table + seq * TABLE_STRIDE + logical, page_valid, 0).to(tl.int64)
    page_valid = page_valid & (physical >= 0) & (physical < BLOCKS)
    d = tl.arange(0, 128)
    k_address = physical[None, :] * KS0 + (j % BLOCK_SIZE)[None, :] * KS1 + kv_head * KS2 + d[:, None]
    k = tl.load(K + k_address, page_valid[None, :], 0)
    if not DEFER_V:
        v_address = physical[:, None] * VS0 + (j % BLOCK_SIZE)[:, None] * VS1 + kv_head * VS2 + d[None, :]
        v = tl.load(V + v_address, page_valid[:, None], 0)
    scores = tl.dot(q, k).to(tl.float32) * scale
    allowed = page_valid[None, :]
    if CAUSAL_BOUNDARY:
        allowed = allowed & valid_rows[:, None] & (j[None, :] <= context + positions[:, None])
    scores = tl.where(allowed, scores, -float("inf"))
    m_new = tl.maximum(m, tl.max(scores, 1))
    safe_max = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(scores - safe_max[:, None])
    alpha = tl.exp(m - safe_max)
    l = l * alpha + tl.sum(p, 1)
    if DEFER_V:
        v_address = physical[:, None] * VS0 + (j % BLOCK_SIZE)[:, None] * VS1 + kv_head * VS2 + d[None, :]
        v = tl.load(V + v_address, page_valid[:, None], 0)
    acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v)
    return m_new, l, acc
```

分页大小 32 的宽 tile 取得两个页号，大页能完整覆盖当前 tile 时广播一个页号，其他布局按 `logical` 逐 lane 查询，随后统一检查物理页在 `[0, BLOCKS)` 内。`KS0/KS1/KS2` 与 `VS0/VS1/VS2` 分别表示物理页、页内 token 和 KV 头的跨度，因此地址计算可以保留真实缓存布局。QK 点积生成 FP32 分数，`m_new` 更新行最大值，`alpha` 把旧质量 `l` 和旧累积值 `acc` 重标定到新最大值后再加入当前 tile，避免直接累加指数导致溢出。`m/l/acc` 保持 FP32，而概率在 PV 点积前经 `p.to(q.dtype)` 转回 Q 的 BF16 类型，尾部因果分支仍按查询位置屏蔽未来 key。`safe_max` 让全空行的指数计算有有限基准，配合初始 `m=-inf、l=0、acc=0` 保持零质量。`DEFER_V` 将 V 的地址与加载放到 QK 和 softmax 后面，这属于源码读取顺序调整，硬件是否形成重叠仍取决于编译和设备执行。

### 2.2 任务映射与运行时数量参数

```python


@triton.jit(do_not_specialize=["TOKENS", "SEQS"])
def iluvatar_paged_gqa_prefill_kernel(
    Q, K, V, O, QStart, SeqLen, Table, Scale,
    TOKENS, SEQS, BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, TABLE_COLS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr,
    OS0: tl.constexpr, OS1: tl.constexpr,
    KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
    VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
    QUERY_TILE: tl.constexpr, SKIP_Q1: tl.constexpr = False,
):
    tile, kv_head = tl.program_id(0), tl.program_id(1)
    # Strictly increasing prefixes also allocate harmless slots for empty queries.
    lo, hi = 0, SEQS
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        prefix = tl.load(QStart + mid) // QUERY_TILE + mid
        if tile >= prefix:
            lo = mid
        else:
            hi = mid
    seq = lo
    q_start = tl.load(QStart + seq).to(tl.int64)
    q_end = tl.load(QStart + seq + 1).to(tl.int64)
    local_tile = tile - (q_start // QUERY_TILE + seq)
    first_query = local_tile * QUERY_TILE
    q_len = q_end - q_start
```

第一维 program ID 表示查询 tile，第二维选择两个 KV 头之一，kernel 先在 `QStart` 累计边界中二分寻找所属序列。搜索值 `QStart[mid] // QUERY_TILE + mid` 加上序列号，使没有查询的序列也占一个安全槽，避免空序列产生相同前缀而混淆归属。找到 `seq` 后，`q_start/q_end` 给出该序列在打包 Q 中的真实范围，`local_tile` 再换算为序列内的 `first_query`。`TOKENS/SEQS` 用运行时标量并标记 `do_not_specialize`，减少数量变化造成的编译体分裂，这个设置只取消这两个数量的特化。页大小、各张量 stride 和查询 tile 仍是编译常量，编译器可以继续按布局生成地址计算，后续有效行检查负责排除额外网格槽。`SKIP_Q1` 的默认值为 False，只有 S03 的已通过兼容检查的 eager Mixed 分支才显式打开，普通 Prefill 和独立调用保持原行为。

```python
    if (first_query < q_len) & ((q_len != 1) | (not SKIP_Q1)):
        M: tl.constexpr = QUERY_TILE * 8
        r = tl.arange(0, M)
        positions = first_query + r // 8
        heads = kv_head * 8 + r % 8
        d = tl.arange(0, 128)
        valid_rows = (positions < q_len) & (q_start + positions < TOKENS)
        q = tl.load(Q + (q_start + positions[:, None]) * QS0 + heads[:, None] * QS1 + d[None, :], valid_rows[:, None], 0)
        seq_len = tl.load(SeqLen + seq).to(tl.int64)
        context = seq_len - q_len
        m = tl.full((M,), -float("inf"), tl.float32)
        # Empty softmax has no mass; do not rely on a first update cancelling
        # an artificial nonzero normalizer initializer.
        l = tl.full((M,), 0.0, tl.float32)
        acc = tl.full((M, 128), 0.0, tl.float32)
        # The first query sees this entire prefix, so all later rows do too.
        full_end = tl.maximum(0, (context + first_query + 1) // 32 * 32)
        stop = tl.minimum(seq_len, context + tl.minimum(first_query + QUERY_TILE, q_len))
        # Q8 reuses wider tiles only in the prefix visible to every query row.
        wide_end = 0
        if QUERY_TILE >= 8:
            wide_end = full_end // 64 * 64
            for begin in range(0, wide_end, 64):
                m, l, acc = _paged_gqa_tile_update(
                    q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                    begin, seq_len, Scale, m, l, acc,
                    BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                    KS0, KS1, KS2, VS0, VS1, VS2, False, 64, True,
                )
```

`r // 8` 选择查询位置，`r % 8` 选择共享同一 KV 头的查询头，因此 Q2/Q8/Q16 分别展开成 16/64/128 行，模型每个 KV 头对应的查询头数始终是 8。首个条件在 Mixed 分派已完成 q1 输出时跳过长度恰为 1 的序列，其余查询继续原 Q16 数学，避免覆盖先前写入的 Decode 结果。Q 从 `q_start+positions` 定位，`valid_rows` 同时检查本序列查询长度和打包 token 容量，越界行加载零且不会写回。`context=seq_len-q_len` 表示当前查询块之前已有的历史，最早查询能看到的完整 32-key 前缀由 `full_end` 截断出来，后续查询也都能看到这段历史。Q8 和 Q16 再把其中完整的 64-key 前缀交给宽 tile，一次读取的 K/V 可供这些查询行和查询头共同计算，减少同一历史数据的重复读取。这个前缀调用关闭逐 query 的 causal mask，因为可见性已由范围证明，`m/l/acc` 仍从空 softmax 状态开始累积。最后不足宽 tile 的共同历史以及查询之间可见性不同的尾部留给下一段处理，宽 tile 不会跨入未来 token；源码首部和此处历史注释中的“eight queries/Q8”描述默认 Q8，当前条件也明确覆盖 Q16。

### 2.3 历史余段、因果边界与输出

```python
        for begin in range(wide_end, full_end, 32):
            m, l, acc = _paged_gqa_tile_update(
                q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                begin, seq_len, Scale, m, l, acc,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, False,
            )
        for begin in range(full_end, stop, 32):
            m, l, acc = _paged_gqa_tile_update(
                q, K, V, Table, seq, kv_head, positions, valid_rows, context,
                begin, seq_len, Scale, m, l, acc,
                BLOCK_SIZE, BLOCKS, TABLE_COLS, TABLE_STRIDE,
                KS0, KS1, KS2, VS0, VS1, VS2, True,
            )
        result = acc / tl.where(l > 0, l, 1.0)[:, None]
        tl.store(O + (q_start + positions[:, None]) * OS0 + heads[:, None] * OS1 + d[None, :], result, valid_rows[:, None])
```

第一个循环从 `wide_end` 到 `full_end` 补齐完整的 32-key 共同历史，继续复用 helper 并关闭逐行因果判断。第二个循环从 `full_end` 到 `stop` 处理每个查询可见范围不同的尾段，helper 的边界分支同时保留真实序列长度和 `j<=context+positions` 判断。比如最后一组查询跨过一个 32-key 边界时，完整历史只计算一次，未来位置仍在尾段被对应查询行屏蔽。宽前缀、窄前缀和尾段连续且不重叠，在线 softmax 的同一组状态贯穿三段，因此每个合法 key 恰好参与一次归一化。最终 `acc/l` 写入调用方 O 的真实 stride 地址，零质量行使用分母 1 得到零，`valid_rows` 保证 padding 和空槽不会覆盖有效输出之外的区域。

```python


def paged_gqa_prefill(query, key_cache, value_cache, output, query_start_loc, seq_lens, block_table, softmax_scale, *, query_tile=8, skip_q1=False):
    """Fill caller-owned output; bridge owns feature/shape compatibility guards."""
    tokens, sequences = query.shape[0], seq_lens.numel()
    if type(query_tile) is not int or query_tile not in (2, 8, 16):
        raise ValueError("query_tile must be 2, 8 or 16")
    if type(skip_q1) is not bool:
        raise ValueError("skip_q1 must be a bool")
    if not sequences or not tokens:
        return output
    iluvatar_paged_gqa_prefill_kernel[(tokens // query_tile + sequences, 2)](
        query, key_cache, value_cache, output, query_start_loc, seq_lens,
        block_table, softmax_scale,
        tokens, sequences, key_cache.shape[0], key_cache.shape[1],
        block_table.shape[1], block_table.stride(0),
        query.stride(0), query.stride(1), output.stride(0), output.stride(1),
        *key_cache.stride()[:3], *value_cache.stride()[:3],
        query_tile, skip_q1, num_warps=8 if query_tile >= 8 else 4, num_stages=1,
    )
    return output
```

Python 入口接收打包 Q、分页 K/V、序列元数据和调用方已分配的 output，先限制 `query_tile` 为整数 2、8 或 16，并要求 `skip_q1` 为真正 bool。没有序列或 token 时直接返回原对象，有数据时用 `tokens // query_tile + sequences` 给每个序列预留足够查询 tile，并在第二维展开两个 KV 头。页表列数、缓存页容量和全部 stride 来自实际张量，kernel 的逻辑位置因此能转换成这些张量的正确物理地址。Q8/Q16 使用 8 warps，Q2 使用 4 warps，三者均为 1 stage；Q16 的实测 shared memory 为 48 KiB、无 spill，兼容性与 fallback 仍由 S03 的 bridge 负责。bridge 在最大查询长度恰为 2 时选择 Q2，在最大查询长度至少 16 且每序列平均实际查询 token 至少 16 时选择 Q16，其余使用 Q8；密度来自整个 batch 的数量，不按评分输入长度选择。`skip_q1` 原样作为编译期开关传入，不增加每层 CPU 扫描、输出分配或压缩查询布局。这里直接填充并返回原 output，避免热路径输出分配和主机读取设备序列长度，也保持调用方对缓冲的所有权。

## 3. 正确性与正式结果

独立 FP32 参照按块表取 K/V，扩展查询头并显式计算 causal softmax 与加权 V，以固定 `atol=0.02、rtol=0.02` 检查 BF16 输出。已覆盖非连续正 stride、分页边界、mixed/empty 查询、常量 V、大值、输出别名、尾部哨兵和输入不变性；Q16 新增局部门包含 15/16/17 附近查询长度、selector/fallback 以及固定底存储下 TOKENS/SEQS 编译体复用。Q16 与 Mixed q1 的恢复对照、全 42 层 builder/bridge/launch/merge 成本及资源门已通过；完整组合的 105 题及本轮两场景正式四轮另见 [S03 第 6 节](S03-天数混合注意力与矩阵访存及配置加载优化.md#6-精度与官方正式结果)。局部 mixed 样例不代表原 C64 请求流程中 q1 的实际分布，分派资格和实际模型阶段解释统一在 S03。

当前组合的完整 105 题精度及正式两场景四轮数据统一在 [S03 正式结果](S03-天数混合注意力与矩阵访存及配置加载优化.md#6-精度与官方正式结果) 维护。该结果对应本文 Prefill、S02 Decode 与 S03 Mixed/GEMM/SQL 的完整组合。
