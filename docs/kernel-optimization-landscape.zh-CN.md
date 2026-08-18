# GPU / TPU Kernel Optimization：现状、空白与切入建议

> 调研日期：2026-08-18。本文把“生成一个更快的单算子”与“在真实系统中稳定地产生收益”分开讨论。由于调研环境无法访问外网，项目细节以公开论文/仓库的稳定接口为锚点；版本号、榜单分数和最新提交没有在本文中声称为已复核事实。

## 1. 结论先行

**不建议再做一个只比较 `PyTorch eager` 与单个 Triton/CUDA kernel 的静态题库。** KernelBench 已经把“从 PyTorch reference 生成正确且更快的 GPU kernel”定义得很好；沿着相同定义扩题，容易变成数据规模竞争，差异化和护城河都弱。

更可行的切入点是：

> **Contextual Kernel Optimization（上下文感知的 kernel 优化）**：输入不只是一个算子，而是来自真实模型的子图、shape 分布、布局/别名、上下游融合机会和硬件预算；输出一个带 dispatch 的 kernel family；用隐藏 workload、冷/热启动、编译成本、显存与跨版本回归共同验收。

它填补的是“microbenchmark 上快”到“生产 workload 端到端快”的鸿沟。建议先做 **Dynamic-shape Kernel Arena**：20–30 个来自 LLM inference、attention、MoE、quantization 和 ragged/batched serving 的小子图，在 2 种 NVIDIA 架构上测量，并从一开始保留 TPU/Pallas backend 接口。

## 2. 现有工作的定位

### 2.1 KernelBench

KernelBench 的核心贡献是把自然的 PyTorch module 变成可自动评测的 kernel 生成任务，并用正确性与相对 PyTorch baseline 的加速共同衡量结果。它适合回答：

- 模型能不能写出可编译、数值正确的 CUDA/Triton 实现；
- 生成实现是否超过统一 baseline；
- 随任务复杂度增加，成功率如何变化。

公开入口：[KernelBench repository](https://github.com/ScalingIntelligence/KernelBench)、[论文](https://arxiv.org/abs/2502.10517)。

**边界**：这类题目的单位通常仍是预先切好的 operator/module；固定或有限 shape、单机单卡、短时 benchmark 更容易测，却不能完整反映编译摊销、动态流量、跨 kernel 中间张量、dispatch、显存峰值和长时间稳定性。

### 2.2 JAX / TPU 路线（本文所称 JAX Bench）

JAX/XLA 的 kernel 优化和 CUDA 路线不完全同构：`jit`、图级编译与异步 dispatch 会显著影响测量；Pallas 暴露 TPU/GPU kernel 编程模型，而 TPU 上还涉及 block spec、VMEM、流水和硬件代际约束。任何 JAX benchmark 若不 `block_until_ready()`、不区分 compile time 与 steady-state time，结果都可能失真。

公开锚点：[JAX benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)、[Pallas documentation](https://docs.jax.dev/en/latest/pallas/index.html)、[JAX repository](https://github.com/jax-ml/jax)。

**与 KernelBench 的互补**：它把问题推向 compiler-in-the-loop 与 accelerator-specific scheduling，但也暴露一个机会——同一高层语义在 CUDA/Triton、JAX/Pallas 和编译器自动生成实现之间，尚缺少以 workload 分布为中心的可移植评测协议。

### 2.3 KernelGYM 类工作

KernelGYM 类环境把 kernel engineering 视为可交互搜索/强化学习问题：agent 获得编译、报错、正确性和性能反馈，然后多轮修改。相较 pass@1，它更接近工程师的工作方式，也适合研究 inference-time scaling、搜索策略、经验库和 reward design。

其典型局限不是“不会搜索”，而是环境 reward 容易被过拟合：

- 针对公开 shape 或单块卡做特化；
- 利用容差、未初始化内存、异步计时或 benchmark 噪声；
- 搜索预算很大，但不计编译与测量成本；
- 在驱动、编译器或硬件变化后收益消失；
- 只优化局部 latency，却增加端到端同步、显存或数值风险。

因此下一代 gym 的价值更可能来自**更真实、难被投机的环境**，而不是再包装一个 agent loop。

## 3. 仍然存在的研究空白

| 空白 | 现有静态单-kernel benchmark 为什么覆盖不足 | 可测量产出 |
|---|---|---|
| 动态 shape 与流量分布 | 单一 shape 鼓励过度特化 | p50/p95/p99、分布加权吞吐、最差 shape 回归 |
| 子图融合与边界选择 | 题目预先决定 kernel 边界 | 端到端延迟、HBM bytes、launch 数、峰值显存 |
| Kernel family + dispatch | “一个实现解决全部输入”不现实 | dispatch overhead、code size、覆盖率 |
| 跨硬件/跨版本稳健性 | 在一块卡上赢不等于可部署 | 多架构 geometric mean、回归率、重编译成功率 |
| 编译成本和冷启动 | steady-state microbenchmark 忽略部署成本 | compile latency、break-even invocation count |
| 数值语义 | 少量随机输入与宽容差容易漏错 | adversarial inputs、ULP/误差分位数、NaN/Inf 语义 |
| 长时间稳定性 | 短跑受频率、缓存与噪声影响 | thermal steady state、方差、错误率 |
| 分布式通信-计算共设计 | 单卡 kernel 不含 collective | 多卡 step time、overlap、网络字节 |
| 可解释性能诊断 | 只有一个 latency reward，学习信号稀疏 | roofline 分类、stall/occupancy、性能归因 |
| 搜索经济性 | pass@k 不等于每美元产出 | best speedup / GPU-hour、time-to-first-win |

## 4. 能否在 compiler pass 的 solution space 上做强化学习？

### 4.1 短答案

**可以，而且 compiler phase ordering 本来就是一个经典搜索问题；但“把所有 pass 任意排列，然后用 RL 搜”通常不是一个好的第一版。** 更准确的研究对象应是：

```text
program state
  --(满足前置条件的语义保持 transformation + 参数)--> new program state
  --(compile / verify / profile)--> constrained reward
```

这里的 action 不仅可以是传统 LLVM pass，还可以是 layout、tiling、fusion、vectorization、software pipelining、memory placement、launch configuration 和 dispatch boundary。真正有价值的空间往往是这些 transformation **及其参数和顺序**的联合空间。

“没有副作用”也需要更严谨：compiler pass 通常承诺在某个语义契约下保持程序含义，但它并不等于数学上的纯函数。浮点 reassociation、原子操作次序、并行 reduction、未定义行为、aliasing、随机数和 race 都可能令两个“合法优化”产生不同 bit pattern，甚至暴露原程序中的未定义行为。因此 correctness verifier 仍然是硬门槛，不能用性能 reward 代替。

### 4.2 三套栈里，哪些部分真的可控？

| 栈 | 类 LLVM pass 的层次 | 外部研究者可控程度 | 更适合作为 RL action 的东西 |
|---|---|---|---|
| CUDA | CUDA C++ 经前端/NVVM IR、PTX，再由 `ptxas` 生成机器码；NVVM 基于 LLVM IR，并有受支持的 IR/编译选项 | **中低**：可以在 LLVM/NVVM 前后做 IR transformation，控制 flags、PTX 和 launch 参数；但 `ptxas` 内部 pass pipeline 并不是一个可任意重排的开放 API | source/IR rewrite、tile/unroll/vector width、shared-memory staging、PTX variant、block size、register/shared-memory budget |
| Triton | Triton dialect → TritonGPU dialect → LLVM/NVVM/ROCDL；编译器内部使用 MLIR pass manager，包含 canonicalization、layout conversion、coalescing、matmul acceleration、pipelining 等 transformation | **高**：开源 compiler 最适合插桩、暴露 pass/attribute，并在每一步做 IR verification；但许多 pass 只在特定 dialect 和 invariant 下合法 | `num_warps`/`num_stages`、tile/layout、CTA decomposition、fusion、pipeline depth、pass 参数与受约束 phase ordering |
| Pallas | Python kernel 被 trace 成 JAX IR，再经 Pallas/Mosaic lowering 进入 TPU 或 GPU 后端；外围还会经过 XLA/JAX compilation | **中**：BlockSpec、grid、memory space、dimension semantics 和部分 compiler params 可控；后端内部 pipeline 的公开可控性不如 Triton，且 TPU/GPU 路径不同 | grid/block mapping、VMEM/HBM placement、pipeline/buffering、dimension semantics、融合边界、backend-specific parameters |

CUDA 的公开锚点是 [NVVM IR specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/) 与 [CUDA compilation workflow](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)。Triton 的 pass 基础可以从其 [compiler source](https://github.com/triton-lang/triton/tree/main/lib) 和 [MLIR pass management](https://mlir.llvm.org/docs/PassManagement/) 观察。Pallas 的可编程界面见 [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html) 与 [Pallas TPU guide](https://docs.jax.dev/en/latest/pallas/tpu/index.html)。

一个重要区别是：**语言有 compiler passes，不代表语言给用户提供了稳定的“枚举全部 passes 并任意排序”的接口。** Triton 最接近可改造的研究平台；CUDA 后端最封闭；Pallas 可搜索的公开空间目前更像 scheduling/configuration space，而不是任意 Mosaic/XLA pass ordering。

### 4.3 为什么排列组合足够大，还不自动意味着 RL 合适？

假设有 30 个 pass、允许长度 20、可以重复，朴素序列空间已经约为 `30^20`；加入 tile size、warps 和 stages 后当然更大。但“大”只说明穷举不可行，并不能证明 RL 优于以下方法：

- autotuning / Bayesian optimization；
- evolutionary search；
- beam search 或 MCTS；
- contextual bandit；
- learned cost model + search；
- imitation learning from compiler/expert trajectories。

RL 值得使用通常还需要满足三个条件：

1. **有可复用的状态信息**：IR graph、shape、target、静态 cost、历史 transformations 能预测下一步收益，而不是每个任务都从零开始。
2. **动作具有长期效应**：某个 pass 当前不加速，却为后续 fusion/vectorization 创造机会；否则 contextual bandit 或普通 autotuner 更简单。
3. **训练能跨程序摊销**：编译与真机测量昂贵。如果只优化一个 kernel，一次性的 evolutionary/Bayesian search 往往更划算；RL 的优势应来自在大量程序和硬件间迁移 policy。

### 4.4 不要搜索任意 pass permutation，要搜索合法 transformation graph

任意排列会产生大量无意义或非法序列：某个 pass 需要特定 dialect；lowering 后不能回到高层 fusion；同一 canonicalizer 连续调用可能无变化；某些 layout transformation 会让后续 pass 的前置条件失效。

更好的环境设计是：

- 每个状态记录 typed IR、dialect、shape/layout、target features、静态资源估计和历史；
- 环境只暴露当前满足 precondition 的 actions，并带参数 mask；
- 每步运行 IR verifier，失败立即终止并给予固定惩罚；
- 使用 IR hash 检测 no-op、cycle 和等价重复状态；
- 高层 transformation 完成后单向 lowering，形成分阶段 DAG，而非任意回跳；
- 将 compiler crash、wrong answer、resource overflow 和 performance regression 分成不同 outcome。

MLIR 的 [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/) 很适合作为这个想法的中间表示：它把“被变换的 payload IR”和“描述变换的 transform IR”分开，并能表达匹配、structured transformation 和失败传播。它比直接让 agent 改 compiler pass list 更容易形成稳定 action schema。

### 4.5 推荐的 RL formulation

**状态**：

```text
IR graph embedding
+ op/dtype/shape/layout features
+ target architecture and compiler version
+ static estimates (bytes, FLOPs, occupancy/resource bounds)
+ applied-transform history
+ previous compiler/profiler feedback
```

**动作采用分层结构**：

1. 选择优化层级：graph fusion、tile/layout、memory、pipeline、lowering、launch/dispatch；
2. 选择当前合法 transformation；
3. 选择离散或有界参数；
4. `STOP_AND_MEASURE`。

**reward 不应只是 `-latency`**。建议：

```text
if wrong_or_unsafe: terminal failure
else reward = log(baseline_latency / candidate_latency)
              - λ_compile * compile_cost
              - λ_search * measurement_cost
```

显存、code size、compile latency 和跨 shape 回归最好作为约束或 Pareto 轴；不要只靠一组容易被投机的权重。真机 latency 只在候选通过静态筛选后测量；中间步骤使用 learned cost model 或 profiler proxy，最终 winner 必须在独立隐藏 workload 上 replay。

**训练/评测必须分开**：按 operator family、模型来源、shape distribution、硬件代际和编译器版本做 held-out split。否则 policy 很容易只是记住“某个公开 kernel 应该用哪个 tile”。

### 4.6 最小可行实验

建议先在 Triton 做，而不是一开始横跨 CUDA、Triton、Pallas：

1. 选 20 个 kernel family，每个 family 提供训练与 held-out shape 分布；
2. 只开放 8–12 个合法 transformation/action，包括 tile、layout、warps、stages、pipeline 和 stop；
3. 建立 random search、evolution、Bayesian/autotuner、beam search 四个强 baseline；
4. RL policy 必须在相同 compile/measure budget 下比较，不比较无限 best-of-N；
5. 评估三件事：time-to-first-win、best speedup/GPU-hour、held-out program/hardware generalization；
6. 成功后再把 action schema映射到 CUDA IR/source rewrites 与 Pallas scheduling knobs。

**Go 条件**：RL 在 held-out kernel family 上、相同真机测量次数内稳定超过 evolutionary/beam baseline，并能把从一代 GPU 学到的 policy 用少量样本适配到另一代。若只在训练 kernels 上赢，或者收益完全来自 `num_warps/num_stages` 调参，那么这仍然只是昂贵的 autotuner，不是通用 pass policy。

### 4.7 有没有人已经在做？相邻工作很多，“Triton pass-policy RL”仍有缝隙

先区分五类容易被混在一起的工作：

| 类别 | 代表工作 | 搜索对象 | 与本文想法的差异 |
|---|---|---|---|
| 通用 compiler phase ordering | CompilerGym、AutoPhase | LLVM flags/pass sequence，或 HLS phase sequence | 已经证明 compiler transformation 可以做 RL；通常不是 TritonGPU layout/pipeline，也不以真机 GPU kernel latency 为主要 reward |
| Learned compiler heuristics | MLGO | inlining、register allocation 等局部决策 | 更强调把 learned policy 嵌入生产 compiler；通常一次学习一个 heuristic，而非端到端搜索一串 Triton transformations |
| Tensor-program schedule search | Halide/TVM learned cost model、Ansor、MetaSchedule | tile、reorder、vectorize、parallel、unroll 等 schedule | 是最强的概念竞争者；常以 evolutionary search + learned cost model 为主，说明 RL 必须在相同测量预算下证明额外价值 |
| Triton 内置 autotuning | `triton.autotune` | 用户提供的一组 `Config`，常见为 tile、warps、stages | 已可解决低维离散配置选择；它不是自动探索 compiler pass sequence，也通常不跨 kernel 学一个 transformation policy |
| GPU binary/instruction optimization | CuAsmRL 类工作 | 编译后的 SASS 指令调度/重写 | 同样使用 RL 和真机 reward，但层次更低；不负责高层 fusion、layout、tiling 和 Triton dialect lowering |
| LLM/agent kernel generation | KernelBench、KernelGYM 类环境 | 生成/修改 Triton 或 CUDA 源码，多轮 compile-and-profile | action 通常是非结构化代码编辑；本文建议的是 compiler-verifiable、带 precondition 的结构化 transformation action |

公开入口包括 [CompilerGym paper](https://arxiv.org/abs/2109.08267) 与 [repository](https://github.com/facebookresearch/CompilerGym)、[AutoPhase](https://arxiv.org/abs/2003.00671)、[MLGO](https://arxiv.org/abs/2101.04808)、[Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)、[TVM MetaSchedule](https://github.com/apache/tvm/tree/main/python/tvm/meta_schedule)、[Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)，以及 [CuAsmRL repository](https://github.com/hgl71964/CuAsmRL)。这些项目的最新维护状态和最新实验数字需要在联网环境中逐项复核。

因此答案不是“没人做”，而是：

- **“用 RL 做 compiler optimization”并不新**，LLVM/HLS 上已有直接先例；
- **“用 learned search 优化 tensor schedule”也很成熟**，TVM/Ansor/MetaSchedule 是必须击败的 baseline；
- **“用 RL 优化 GPU 机器码”也已有相邻工作**；
- 相对更窄、仍可能有贡献的命题是：**在 Triton 多层 IR 上定义可验证的 transformation graph，学习能跨 kernel family、shape 和 GPU 架构迁移的 policy，并在相同 compile/measure budget 下优于 autotuner、evolution 和 cost-model search。**

这个表述非常重要。若论文标题只是“用 PPO 选择 `num_warps`、`num_stages` 和 tile size”，新颖性会很弱，因为它本质上是在替换 autotuner 的搜索算法。更有辨识度的贡献应至少占到下面三项中的两项：

1. **新的 action abstraction**：跨 Triton IR/TritonGPU IR 的合法、可组合 transformation schema；
2. **新的 generalization result**：held-out operator family、动态 shape 或跨 GPU 架构迁移，而非 per-kernel 从零搜索；
3. **新的 systems result**：在固定 GPU-hour、compile 次数和测量次数下，显著改善 time-to-first-win 或 Pareto frontier；
4. **新的数据资产**：公开完整 transformation trajectory、失败类型、IR snapshot 和 profiler feedback；
5. **新的 verifier**：证明 policy 无法靠公开 shapes、计时漏洞或错误数值语义投机。

### 4.8 最值得先做的 novelty check

在正式训练 RL 前，用一周做下面的“去新颖性”实验：

1. 从 Triton compiler 选 6–8 个 transformations/knobs，写出显式 precondition 和参数域；
2. 对 5 个 kernel family 穷举可承受的短序列，确认 pass 顺序确实产生 interaction，而不是最后只剩独立超参数；
3. 比较默认 Triton pipeline、`triton.autotune`、random、evolution 和 beam search；
4. 检查最优序列是否能迁移到 held-out shapes；
5. 若顺序交互很弱，把课题改成 learned cost model/dispatch；若交互很强且存在跨任务规律，再投入 RL。

最关键的证据不是 solution space 的理论大小，而是 **action interaction**：需要展示 `A → B` 明显优于 `B → A`，且这种规律能跨任务复用。没有这一点，MDP 会退化成昂贵的组合调参问题。

## 5. 候选方向排序

评分：5 为最好；“难度”5 表示最难。

| 方向 | 新颖性 | 用户价值 | 可在 8–12 周验证 | 防御性 | 难度 | 建议 |
|---|---:|---:|---:|---:|---:|---|
| 动态 shape + 子图 + dispatch arena | 4 | 5 | 4 | 4 | 4 | **首选** |
| adversarial verifier / anti-cheating suite | 4 | 5 | 5 | 4 | 3 | 与首选绑定 |
| 跨 GPU/TPU performance portability | 5 | 4 | 2 | 5 | 5 | 第二阶段 |
| 分布式通信-计算 kernel co-design | 5 | 5 | 2 | 5 | 5 | 有集群资源再做 |
| 性能数据/轨迹与 learned cost model | 3 | 4 | 3 | 5 | 4 | 可成为数据护城河 |
| 再做一套静态 PyTorch-to-Triton 题库 | 2 | 3 | 5 | 2 | 2 | 不建议单独做 |

## 6. 推荐产品：KernelRace Arena

### 6.1 Task contract

每道题不再只给 `(reference_fn, example_inputs)`，而给：

```text
semantic reference
+ allowed dtype / error contract
+ hidden shape-and-layout distribution
+ aliasing / mutation contract
+ target hardware set
+ cold-start and memory budget
+ optimization boundary (op, subgraph, or pipeline stage)
```

提交物是可复现构建的 `kernel family + dispatcher`，而非一个源码字符串。dispatcher 不能读取隐藏标签，只能使用运行时合法可见的信息（shape、stride、dtype、device capability）。

### 6.2 分层评分

先设硬门槛，再比较 Pareto 指标，避免一个含糊的加权总分掩盖失败：

1. **Correctness gate**：随机、边界、对抗输入全部通过；检查 NaN/Inf、非连续 tensor、别名及 deterministic contract。
2. **Safety gate**：compute-sanitizer/等价工具、越界 canary、重复运行和多 stream 测试通过。
3. **Performance**：对隐藏请求分布计算端到端 latency 的 geometric mean，同时报告 p50/p95/p99 与 worst-shape regression。
4. **Economics**：报告编译时间、搜索 GPU-hours、首次超过 baseline 的时间，以及按真实调用次数计算的 break-even point。
5. **Portability**：按硬件分别出分；可另给跨硬件 harmonic/geometric mean，但绝不能用平均值隐藏某一硬件的失败。

可把主分数定义为：

```text
score = correctness_gate × safety_gate ×
        geo_mean_hardware(E_workload[baseline_latency / candidate_latency])
```

编译成本、峰值显存和搜索成本作为约束与独立 Pareto 轴，而不是任意系数混入主分。

### 6.3 首批任务

优先选择“shape/融合决定比语法翻译更重要”的任务：

- RMSNorm + residual + quantize；
- RoPE + KV-cache update（paged/ragged layouts）；
- grouped-query attention 的 decode 小 batch 分布；
- MoE routing + permute/unpermute + grouped GEMM 周边；
- dequantize + matmul + bias/activation；
- variable-length segment reduction / scatter-gather；
- optimizer update with mixed precision；
- 具有生产 shape 直方图的 embedding / sampling 子图。

任务来源必须保留 provenance、license 和去重指纹；测试 shape 不应直接进入公开训练集。

## 7. 8–12 周 MVP

### 第 1–2 周：协议与可靠 harness

- 定义 task schema、submission ABI 和容差契约；
- fork/适配成熟 benchmark timing 方法，不自行发明一次性计时器；
- 建立 eager/framework、compiler-generated 和 vendor library 三类 baseline；
- 固定 GPU 时钟策略或至少记录 clocks、temperature、driver、compiler 与完整环境。

### 第 3–5 周：10 个任务与隐藏分布

- 从 2 个真实开源模型/serving pipeline 提取 10 个子图；
- 每题建立 public smoke shapes、private validation shapes、held-out distribution；
- 用 metamorphic properties 辅助验证，例如 permutation、scale、分块等价关系。

### 第 6–8 周：agent loop 与分析

- 接入任一代码 agent，统一限定 1/10/100 次编译预算；
- 保存每一步 source、compiler log、correctness、latency 和 profiler 摘要；
- 对比 pass@1、best-of-N、带性能记忆的搜索，以及人工专家 baseline。

### 第 9–12 周：多硬件复现与发布

- 至少覆盖两代 NVIDIA GPU；若 TPU 暂不可得，发布 backend interface 和 Pallas prototype，而不要虚构跨 TPU 结论；
- 预注册 scoring protocol，冻结隐藏测试；
- 发布 leaderboard、reproducer container、原始测量分布与失败分类。

**MVP 成功标准**：

- 同一候选在公开 shape 赢、隐藏流量输的现象能被稳定测出；
- 重复测量的排序稳定，关键任务 95% 置信区间足够窄；
- 至少 3 个任务中，最优方案需要多实现 dispatch 或融合边界选择；
- 可以报告“单位搜索成本带来的收益”，而不只报告 best-of-N；
- 另一台同型号机器能复现实质结论。

## 8. 实验设计中必须避免的坑

1. **异步计时**：JAX 使用 `block_until_ready()`；CUDA 使用 events 并正确同步。
2. **baseline 失真**：同时比较 framework eager、`torch.compile`/XLA、vendor libraries；否则只是在打一个弱 baseline。
3. **输入泄漏**：公开 smoke tests 与私有分布分离；定期轮换 held-out workload。
4. **只测平均数**：保存原始 samples，报告置信区间、尾延迟和回归 shape。
5. **错误容差单一**：误差契约按运算和 dtype 定义，不能全局套 `rtol/atol`。
6. **缓存混淆**：明确 warm cache/cold cache、compile cache、autotune cache 的状态。
7. **忽略 side effects**：检查输入未被非法修改、stream 语义、allocator 与 graph capture 兼容性。
8. **硬件不可比**：不把不同功耗上限、时钟和共享机器负载下的结果放在同一榜单。
9. **只报赢家**：保留所有搜索尝试，计算选择偏差并在独立 replay set 复测 winner。
10. **训练污染**：记录题目公开时间、哈希与相似代码检索结果，区分 memorization 与 generalization。

## 9. 真正可形成护城河的资产

代码框架本身容易复制。更难复制的是：

- 经脱敏且有代表性的 shape/layout/arrival 分布；
- 跨硬件、跨编译器版本的完整优化轨迹；
- 能定位错误与性能退化原因的 verifier/profiler 数据；
- submission 在后续驱动版本上的持续 replay 历史；
- 从失败轨迹训练的 cost model、retriever 和 repair policy。

因此商业上可采用“开放协议与小型公开集 + 私有持续更新 workload + 托管多硬件评测”；研究上则发布足够的冻结集和原始数据，保证结论可复现。

## 10. Go / No-Go 决策

在投入 TPU 或大规模集群前，先做一个两周 spike：选 3 个动态任务、2 张不同代际 GPU、3 个 baseline。若满足以下任意两项，继续：

- 单 shape 排名与 workload-distribution 排名发生反转；
- 最优单 kernel 被小型 dispatcher 稳定击败；
- `torch.compile`/vendor baseline 与 eager baseline 的结论明显不同；
- 候选在另一代 GPU 出现超过 20% 的相对回归；
- 计入编译成本后，最优解的 break-even 超出实际服务生命周期。

若完全观察不到这些现象，说明所选任务仍过于接近静态 microbenchmark，应更换任务，而不是扩大题量。

## 11. 参考资料

- Scaling Intelligence, [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench).
- Ouyang et al., [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517).
- JAX, [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html).
- JAX, [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html).
- OpenAI, [Triton language and compiler](https://github.com/triton-lang/triton).
- NVIDIA, [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/).
- NVIDIA, [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/).
- NVIDIA, [NVVM IR specification](https://docs.nvidia.com/cuda/nvvm-ir-spec/).
- LLVM/MLIR, [Pass management](https://mlir.llvm.org/docs/PassManagement/) and [Transform dialect](https://mlir.llvm.org/docs/Dialects/Transform/).
- Triton, [Compiler source](https://github.com/triton-lang/triton/tree/main/lib).
- JAX, [Pallas design](https://docs.jax.dev/en/latest/pallas/design/design.html).
- Meta, [CompilerGym](https://github.com/facebookresearch/CompilerGym).
- Haj-Ali et al., [AutoPhase](https://arxiv.org/abs/2003.00671).
- Google, [MLGO](https://arxiv.org/abs/2101.04808).
- Zheng et al., [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng).
- Apache TVM, [MetaSchedule](https://github.com/apache/tvm/tree/main/python/tvm/meta_schedule).
- Triton, [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html).
- [CuAsmRL](https://github.com/hgl71964/CuAsmRL).
- MLCommons, [MLPerf Inference policies and methodology](https://github.com/mlcommons/inference_policies)（可借鉴系统级提交、审计与复现原则，不代表其直接评测 kernel generation）。
