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

## 4. 候选方向排序

评分：5 为最好；“难度”5 表示最难。

| 方向 | 新颖性 | 用户价值 | 可在 8–12 周验证 | 防御性 | 难度 | 建议 |
|---|---:|---:|---:|---:|---:|---|
| 动态 shape + 子图 + dispatch arena | 4 | 5 | 4 | 4 | 4 | **首选** |
| adversarial verifier / anti-cheating suite | 4 | 5 | 5 | 4 | 3 | 与首选绑定 |
| 跨 GPU/TPU performance portability | 5 | 4 | 2 | 5 | 5 | 第二阶段 |
| 分布式通信-计算 kernel co-design | 5 | 5 | 2 | 5 | 5 | 有集群资源再做 |
| 性能数据/轨迹与 learned cost model | 3 | 4 | 3 | 5 | 4 | 可成为数据护城河 |
| 再做一套静态 PyTorch-to-Triton 题库 | 2 | 3 | 5 | 2 | 2 | 不建议单独做 |

## 5. 推荐产品：KernelRace Arena

### 5.1 Task contract

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

### 5.2 分层评分

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

### 5.3 首批任务

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

## 6. 8–12 周 MVP

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

## 7. 实验设计中必须避免的坑

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

## 8. 真正可形成护城河的资产

代码框架本身容易复制。更难复制的是：

- 经脱敏且有代表性的 shape/layout/arrival 分布；
- 跨硬件、跨编译器版本的完整优化轨迹；
- 能定位错误与性能退化原因的 verifier/profiler 数据；
- submission 在后续驱动版本上的持续 replay 历史；
- 从失败轨迹训练的 cost model、retriever 和 repair policy。

因此商业上可采用“开放协议与小型公开集 + 私有持续更新 workload + 托管多硬件评测”；研究上则发布足够的冻结集和原始数据，保证结论可复现。

## 9. Go / No-Go 决策

在投入 TPU 或大规模集群前，先做一个两周 spike：选 3 个动态任务、2 张不同代际 GPU、3 个 baseline。若满足以下任意两项，继续：

- 单 shape 排名与 workload-distribution 排名发生反转；
- 最优单 kernel 被小型 dispatcher 稳定击败；
- `torch.compile`/vendor baseline 与 eager baseline 的结论明显不同；
- 候选在另一代 GPU 出现超过 20% 的相对回归；
- 计入编译成本后，最优解的 break-even 超出实际服务生命周期。

若完全观察不到这些现象，说明所选任务仍过于接近静态 microbenchmark，应更换任务，而不是扩大题量。

## 10. 参考资料

- Scaling Intelligence, [KernelBench repository](https://github.com/ScalingIntelligence/KernelBench).
- Ouyang et al., [KernelBench: Can LLMs Write Efficient GPU Kernels?](https://arxiv.org/abs/2502.10517).
- JAX, [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html).
- JAX, [Pallas: a JAX kernel language](https://docs.jax.dev/en/latest/pallas/index.html).
- OpenAI, [Triton language and compiler](https://github.com/triton-lang/triton).
- NVIDIA, [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/).
- NVIDIA, [Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/).
- MLCommons, [MLPerf Inference policies and methodology](https://github.com/mlcommons/inference_policies)（可借鉴系统级提交、审计与复现原则，不代表其直接评测 kernel generation）。
