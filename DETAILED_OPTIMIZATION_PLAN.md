# MiniCPM5-2B 双平台推理吞吐优化路线

## 1. 当前重点

当前仅推进官方 OnlineLab 天数 BI-V150（32 GiB），MetaX 暂停。canonical 已纳入冻结的 Mixed/runtime/GEMM/SQL 组合，双场景正式四轮已通过，改后实际模型短Profile已完成。模型、原 serve 参数、原 wrapper、数值库、kernel/config 和后端选择保持不变；vLLM 只读，已授权本地提交；无推送、续租或实例关机授权。

当前正式B1的原wrapper双场景各四轮完整通过，唯一A0/B1正式四行表、完整精度摘要与统计口径位于 [B1-V150 S03 第 6 节](docs/B1-V150/S03-天数混合注意力与矩阵访存及配置加载优化.md#6-精度与官方正式结果)。完整实现按唯一归属维护：[S01 Prefill](docs/B1-V150/S01-天数分页GQA注意力优化.md)、S02 Decode、[S03 Mixed/GEMM/SQL](docs/B1-V150/S03-天数混合注意力与矩阵访存及配置加载优化.md)，当前正式统计如下。

本轮数值实现已通过Mixed全成本、Decode runtime编译体复用及实际GEMM四shape净收益/恢复对照门。七数值源码组合的完整Level3于14:58:07–15:23:02完成：105唯一预测/review、105非空HTTP200、失败0、103/105正确（98.095238%）。后加SQL只改变CPU反射，不改变数值七源及实际kernel/config；因此复用本组合103/105，SQL没有另跑fresh105。原CLI开发门已完成并完整归档，不作为正式统计。

实际模型bounded嵌套CPU计时确认54次index内部反射合104.369 s；仍有配对的17次外层调用中反射占81.66%–90.31%，另11个cached长窗没有反射，不把全部CPU耗时归SQL。仅对iluvatar现有默认内部SQLite的同connection显式BEGIN内全库无FK/无未知对象资格和exact callable only反射作变更，保持默认FK解析、finally rollback/close和fresh原full fallback。只读资格/等价/净省时三门及生产自身实际import门通过，实际DB474表、474索引、10240000 bytes且字节不变；生产自身三表full→guardedonly→restore为1783.24→10.42→1746.35、1806.34→10.84→1905.73、1858.62→10.00→1994.65 ms。不改get_sql_model/key/pool/lock/puts/session/tuner/cache。

八源冻结组合于16:48:34–18:06:22完成原wrapper RUNS=4、SKIP_FIRST=1双场景8轮，退出码0。4K四轮raw total3716.04/3714.91/3743.46/3799.76，后三轮3752.71±43.17 tok/s、CV1.1505%、Mean TTFT4366.20 ms；每轮256/256成功、失败0、输入1048576/输出262144。16K四轮raw2952.21/2951.79/2952.79/2951.60，后三轮2952.06±0.64 tok/s、CV0.021656%、Mean TTFT170295.23 ms；每轮128/128成功、失败0、输入2097152/输出131072。逐场景相对同日accepted Q16 total提升3.2079%/3.9769%、Mean TTFT下降10.6193%/4.9435%，相对官方A0提升85.0440%/222.5766%。8轮请求/tokens、吞吐和TTFT门全部通过，服务ERROR/OOM及cgroup failcnt/oom_kill/under_oom均0，preemption累计16，采样显存峰28688 MiB。完整raw/summary/controller/monitor已同步，root已审核接受为当前天数B1，五生产文件已用apply_patch纳入canonical，S01/S02/S03与索引已更新，无commit/push，评分snapshot八源不变。改后同源actual-model Profile于18:10:16–18:29:55完成两个原cohort，均exit0，共七窗口；自有评分与Profile进程均已严格按身份停止，GPU任务结束。远端现有trace解析已完成，4K初始/heavyMixed GEMM56.46%/54.08%、Prefill32.57%/31.78%，heavy q1 partial/merge4.43%/0.19%；16K初始Prefill64.13%、GEMM30.70%。稳态及tail每窗8次FULL graphlaunch，图内保持opaque；当前后续工作按下述同步与验证预算推进。

当前工作顺序：

1. 本轮正式双场景门已完成并完整同步，由root只读审核；不重做环境摸底、源码哈希或4K测试，不重复正式轮。
2. 正式按后三轮mean、sample SD和CV逐场景比较同日accepted Q16，两个场景均无稳定回退；全部raw、请求/tokens、TTFT、ERROR/OOM、显存及preemption完整保留。
3. 改后同源非评分actual-model七短窗已完成，两个原cohort成功、ERROR/OOM0；自有进程均已按身份停止。FULL图内未展开部分保持opaque，阶段事实已原位写入S03。
4. 已同步获准生产文件至canonical并更新S01/S02/S03及索引；全部生产性能更改代码连续收录一次、块不超过40行并解释，唯一正式四行表与精度、统计说明维护于最新编号S03末尾，旧页只保留结果链接。Profile事实已更新，代码唯一覆盖和正式表已通过root只读核查。
5. 当前正式B1继续保留，五生产文件与正式八源保持同版。同源SDK定时设备诊断完成原4K 256请求、失败0、输入1048576/输出262144，client及collector均exit0；同期metrics支持稳定64请求Decode。首8个完整连续replay设备区间495.858802 ms，FULL Graph aggregate为427.860848 ms（86.2868%），图外topk/topp为48.391880 ms（9.7592%），linear为8.995046 ms（1.8140%）；图内算子仍为opaque aggregate。实际sampler入口取得稳态B64×130560 FP32过滤前快照，k=None、p=0.95；同一快照的位模式、RNG状态及完整调用成本记录已归档。本轮隔离数值与完整调用成本检查已结束，原始结果已同步；当前保留正式B1，未增加生产实现，自有GPU任务已结束。
6. MetaX暂停至用户恢复，保留 [MetaX S01](docs/MetaX/S01-C500基线运行与内存问题复现.md)和 [MetaX S02](docs/MetaX/S02-C500排序索引优化.md) 的既有实现与结果。

本轮截止北京时间2026-10-12 24:00，23:40最终同步；之后只同步与按身份精确停止自有进程，不自行续租、实例关机或提交。正式双平台出口、干净复现及适用路径/fallback回归留待后续阶段。

## 2. 统一约束与验收

本机只做源码、文档和 Git 静态工作，目标卡才运行导入、编译、测试、服务、Profile 与 benchmark。指定版本、规则问题和调用链统一引用 [S01](docs/S01-赛前静态准备.md)；实验与文档纪律遵循[项目 skill](skills/flagos-minicpm-competition/SKILL.md)。只修改 `vllm-plugin-FL` 和 `FlagGems`，`vllm` 保持只读参考。

必须保持官方 benchmark、test cases、serve 参数、模型、采样、输出长度和评分输入不变。禁止通过超参数开启量化或投机采样，禁止调 benchmark，禁止只切换现成 operator 或删除主要选择逻辑取得性能，禁止直接合并最新推理框架分支；同时不得启用现成前缀缓存或按 `4096/16384`、case ID 写死分支。

每项成果必须包含可审查的实质计算或访存代码，报告精确映射实际激活路径，并为未知 shape、layout、dtype 和平台保留原实现 fallback。不得为命中候选而切换既有 backend；租赁结果只称预验证，规则解释和正式结论以比赛页面、组委会书面答复及官方平台为准。

官方A0与门槛如下；A0始终只指组委会公布值。吞吐有效性下限为A0×0.99；A0×1.01是涨点>1%的参考线，不是官方否决阈值，稳定收益还须超过实测噪声：

| 平台 | 场景 | A0 total tok/s | 官方下限A0×0.99 | 涨点参考A0×1.01 | A0 Mean TTFT | TTFT 上限 |
|---|---:|---:|---:|---:|---:|---:|
| BI-V150 | 4K/1K | 2028.01 | 2007.73 | 2048.29 | 11573.53 ms | 11689.27 ms |
| BI-V150 | 16K/1K | 915.15 | 906.00 | 924.30 | 599262.03 ms | 605254.65 ms |
| C500 | 4K/1K | 5089.645 | 5038.75 | 5140.54 | 3199.435 ms | 3231.43 ms |
| C500 | 16K/1K | 7029.675 | 6959.38 | 7099.97 | 27197.135 ms | 27469.11 ms |

所有候选采用同一验收链：

1. 先做算子差分、边界 shape、layout 和 fallback 测试；任何输出、token、EOS、长度或请求错误都会使性能结果作废。

2. 数值路径改变的单项或同主线组合候选通过局部差分后，统一运行完整 MATH-500 Level 3，至少 `100/105` 正确，并保存逐题输出、采样身份和失败请求；不为组合内每个小改动重复完整评测。

3. 在同一目标平台与不变服务参数下，用必要的短配对、恢复 B1 对照或消融排查漂移与交互；对待验收的最终组合统一运行官方 4K/16K wrapper，各四轮、丢弃第一轮，报告后三轮 mean、SD、CV 并保留所有轮次，包括慢轮。不要求组合内每个子项重复三臂全程，最终组合的稳定收益与完整验收门不变。

4. 要求所有请求成功、无 OOM，Mean TTFT 不越表中上限，两个场景的吞吐收益均超过实测噪声；最后补改后 Profile，并在另一平台执行适用路径与 fallback 回归。

5. 只有 microbenchmark 收益、路径未激活、无法映射最终 diff 或端到端落入噪声的补丁不保留。相关修改积累成完成正确性、性能和保留决定的阶段成果后再批量提交，除非用户另行授权。

每次远端实验结束后立即同步原始结果和源码身份，并原位更新相关学习文档；沿用现有 SNN，不为一次小实验新建文档或小提交。每个平台最高编号现存 SNN 末尾维护唯一 A0/B1 四行表、完整精度摘要与统计口径；依既有创建门槛或用户明确许可新建专题后立即整段迁移而非复制，旧页只保留链接，索引与计划同步。迁移不改变数值、接受状态或租赁预验证边界。

## 3. 七阶段实施

### 阶段 1：规则、环境与可信基线

**问题与目标。** 容器实际 import、设备 backend 和仓库 checkout 可能不同；先让双平台代码身份、模型行为、精度和波动可追溯。

1. 按 S01 确认规则与调用链，记录设备、驱动、镜像、Python、Torch、编译器、plugin、FlagGems、实际 import 路径和源码状态。

2. 从模型类、注册、dispatch、编译实现到设备 kernel 核实实际链路。KV update 的 C0 只在 BI-V150 真正命中 `AttentionFLBackend` 时检查，否则记录不适用并分析真实 backend。

3. 运行确定性短、长和混合 prompt smoke，核对 token、EOS、生成长度、失败请求和重复异常，再执行完整 Level 3。

4. 两个平台的 4K/16K 基线至少独立重复三组，保存四轮原始数据与后三轮统计，用于识别约 1% 的变化。

**验收。** 实际 import/backend 可追溯，适用时 C0 正确，smoke 和 Level 3 通过，双平台基线波动足以判断约 1% 收益。

### 阶段 2：分阶段 Profile 与瓶颈归因

**问题与目标。** 总吞吐无法区分 Prefill、Decode、CPU 空洞或内存压力；需用联合测量解释至少 85% 端到端时间并冻结真实热点顺序。

1. 在两平台、两场景分别标记纯 Prefill、mixed、稳态 Decode 和 tail，请求阶段必须由 token 与 annotation 共同确认。

2. 同步采集 CPU timeline、device kernel、内存水位、allocation/copy、同步、graph capture/replay 和 break，不只截取最快片段。

3. 将时间归入 GEMM、Attention、RoPE、KV read/write、Norm/激活、metadata/scheduler 与空洞，并核实每个热点对应的实际源码路径。

4. 只用匹配时间窗的占比计算 Amdahl 上限；收益不足以越过噪声的候选暂缓，每个平台冻结前三项实际热点。

**验收。** 四个正式场景都有分相热点与收益上限，至少 85% 端到端时间可解释，候选顺序由 trace 而非源码外观决定。

### 阶段 3：低风险同步与分配清理

**问题与目标。** `M1` 处理 C500 非 cascade、DCP=1 且有 Prefill 的路径；其余同步、分配和 wrapper 只有被 Profile 证明可见才处理。

1. 在 metadata builder 一次构造并共享 `prefill_cu_seq_lens`，供 42 层使用，不在每层 `.tolist()` 后重建 tensor。

2. 对 dummy tensor、descale、view/chunk、`.to()` 和固定 buffer 分别核实读取关系、allocation/copy 与地址需求，不直接删除接口必需对象。

3. 仅对已测得的开销做单变量修改，并为 mixed batch、非连续布局、量化 KV、未知 shape/device 保留边界 fallback。

4. 用目标计数器、token 差分、Level 3 和两场景配对验证；源码更短但计数器与端到端不变的补丁删除。

**验收。** 目标同步或分配按设计消失，metadata 语义和 graph 地址正确，4K/16K 无稳定回退；当前租赁 C500 结果仍待官方双平台出口。

### 阶段 4：BI-V150 Attention 与 KV

**问题与目标。** 当前 Q16 组合的分页 Prefill、直接序列索引 Decode 与数量参数编译策略已完成局部数学、原Decode动态图、本版完整105及天数双场景正式四轮。正式结果统一在天数S03第6节；同源actual-model分相Profile已完成初始、稳态和tail短窗，FULL内部未展开部分如实保留；当前八源组合正式4+4已通过并纳入canonical，改后Profile已完成，跨平台适用路径回归留待后续阶段。

1. **优先级 1：P1 自研 paged-GQA Prefill/mixed Attention。** FlagGems 的 iluvatar 实现已组合跨 query KV 复用、完整历史/因果边界拆分、页粒度寻址与冗余 mask 削减、Q2/Q8/Q16 选择四项工作；`max_query_len == 2` 优先 Q2，否则只有最大查询至少16且整个batch平均查询至少16才选Q16，其余Q8。保持 BF16、FP32 online softmax、原 scale/causal 语义与输出 buffer contract，保留 physical page 负值/越界保护、Q 读取与输出写入 mask。plugin 仅通过原 `TRITON_ATTN` 枚举的受限实现子类接入，保留原 selector、KV update、mixed FULL graph 与未知 shape/layout/feature 的 fallback，不修改 vLLM，不启用现成 FlagGems Attention 开关。

2. **优先级2：当前 Mixed/runtime/GEMM/SQL 组合正式保留。** Q8/Q16 的完整共同历史使用64-key，余32/因果/Q2保持原分段；Decode按设备长度选择32或64，固定16段与三个FP32 workspace。Prefill数量参数为runtime且禁止按值特化，数学、布局与兼容guard保持不变。本版完整105及原wrapper双场景各四轮门已通过，已保留四份数值生产文件及一份CPU SQL文件，未提交。

3. **当前Mixed与后续稳态Decode研究。** 实际高q1 Mixed及自身KV长度和Attention设备热点已经支持原16段Decode/q1与Q16 skip的三文件关联包，builder及全部launch/merge成本已纳入局部门，最终组合已通过天数正式双场景验收。后续同源非评分Profile保持FULL内部opaque边界，不切graph/后端求可见性。仅在真实稳态长度分布及partial/merge设备计时证明工作失衡后，才研究固定CTA队列思路；现阶段不实施该路线，不扩大当前冻结源码。

4. paged-KV 读取已计入 Attention 主核，不能重复相加。实际 mixed 窗口中的独立 KV write 不是首要热点；后续只有实际占比或可消除访存量支持时才研究 KV/RoPE 融合，并保持 slot mapping、负 slot、跨 block、cache alias 与未写区域语义。

5. C500 `M1` 针对的逐层 CPU `.tolist()` 路径在当前 iluvatar Attention 中不存在，不照搬补丁。Norm/SiLU 等已有融合先沿用，只有 BI 实际热点证明还有可消除工作时才继续开发，不重复宣称已有融合；sort/sampler 差分微测未见可分辨回归，本轮不继续排序。双平台沿现有 dispatch、平台 guard 和原实现 fallback 共用同一合并源码，不重造兼容框架或强行统一 backend。

6. 调度、graph 策略与自研量化留待规则许可、适用审批和真实热点支持后最后评估。BI 的 32 GiB 与 KV 压力不是修改正式 `gpu-memory-utilization`、`max-model-len`、KV cache dtype、并发或采样参数的理由；不通过服务配置、量化开关或更换现成 backend 求分。

**验收。** 设备路径实际激活，Attention/KV 差分、必要的完整 105 题精度、BI 4K/16K 官方四轮及跨平台适用路径/fallback 门通过，吞吐提升超过噪声，TTFT、请求正确性与 shadow shape 不回退。当前八源组合的局部正确性、103/105数值精度和天数正式双场景四轮已完成；改后Profile已完成，双平台门仍待后续阶段。

### 阶段 5：C500 Attention、KV 与融合主线

**问题与目标。** 当前 C500 主热点是 MACA Attention，但安装版 `flash_attn` 的签名、binding 和可提交源码边界未确认；本阶段必须交付至少一项非排序、实际激活的实质代码，不能用已完成的排序成果代替。

1. 在目标容器核实 MACA 的 actual import、函数签名、编译 binding 与源码权限，并分别 Profile 主核、combine、copy、KV write 和 CPU 空洞。实际 Decode 接口未证明支持 `num_splits` 或 direct output 前，不把 metadata 字段当调优旋钮。

2. 首攻 paged-KV 读取、GQA 跨 Q head 的 K/V 复用和 split/combine 工作分配，保持 BF16 输入输出、FP32 softmax 状态、scale 与 causal 语义。

3. 若 MACA 主核在不可编辑边界外，评估 FlagGems 自主受限 BF16 paged-GQA Decode kernel；只能从 plugin 原调用点做 scoped dispatch，并保留原 backend selection 和 fallback，不能仅切换现成实现。

4. KV/RoPE 融合先过成本门，核实 slot mapping、cache layout、ownership、alias 与可消访存量；保持 Q 旋转、K 原地语义和 cache contract，避免重复写 cache。独立 KV write 占比很小，不能单独支撑融合。

5. Norm/SiLU 当前已有 Inductor 融合，QKV 与 gate/up 也已合并；不得重复宣称这些融合，但新的 Profile 若证明已激活 kernel 仍有计算或访存瓶颈，可以做实质优化。graph 变更留待阶段 6 规则门。

**验收。** 至少一项 Attention、KV 或融合的非排序代码在 4K/16K 都独立稳定生效；覆盖 mixed、2K/8K/12K、head mapping、paging 边界、layout、workspace 与 fallback，并通过统一 Level 3、配对 wrapper 和改后 Profile。租赁结果不代替官方双平台验收。

### 阶段 6：通用调度、图执行与可选量化

**问题与目标。** 调度和 graph 容易变成 benchmark 特调，量化又有较高精度与规则风险；只有书面规则允许且前五阶段稳定后才进入。

1. 先取得 scheduler、graph 和自研量化边界的书面确认，不改变正式 serve 参数。

2. 调度只使用 phase、活动序列、当前 token、token budget、最大 KV 长度、KV 水位和 graph eligibility，不读取 case 名或固定长度。

3. 提高 graph capture/replay 覆盖时保留动态 shape 与不支持路径的 eager fallback，并用 2K/8K/12K、mixed 和 tail 检查泛化。

4. 自研 W8A16 只在规则允许且权重带宽为热点时最后评估，先做单层差分再整模精度；不得通过超参数开启量化、投机采样，W4 与 KV 量化不作为早期候选。

**验收。** 四个正式场景和 shadow shapes 使用同一语义策略，无硬编码、饥饿、OOM 或 TTFT 越线；graph fallback 可解释，量化若启用则完整 Level 3 与净端到端收益均通过。

### 阶段 7：组合、复现与提交

**问题与目标。** 单项通过不代表组合有效；最终只组合已验收补丁，并让正式数据、最终 diff 和技术报告一一对应。

1. 按单项验收顺序组合补丁，每加入一项都做消融、短 prompt、算子差分和快速配对，出现交互时回到最后一个已知正确组合。

2. 在 BI-V150 与 C500 分别运行完整 Level 3、4K/16K 官方流程和全精度回归，整套正式流程独立复现至少两次。

3. 从第二个干净 checkout 重放补丁，核对 backend、实际 import、命令、精度、性能和 fallback。

4. 报告逐项映射“问题分析→实际代码→正确性→性能→规则依据”，保留完整消融，不用单次峰值掩盖场景回退。

5. 删除未激活、无收益、仅 microbenchmark、诊断开关、case 硬编码和报告无法解释的路径，再整理提交包。

**验收。** 两平台、两场景均无失败请求并稳定超过噪声，完整精度与 TTFT 合规；两次独立复现和第二 checkout 结果一致，最终报告精确对应提交代码。
