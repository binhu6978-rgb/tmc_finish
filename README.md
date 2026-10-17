# SemConsFL Semantic Utility Probe

## 最新任务：B0/B1各300轮Full验证峰值

在原项目上增量加入两个300轮入口，旧Original/Fixed-Class 50轮诊断保留为历史版本。

- **B0**：联合task CE＋semantic CE，记录旧Anchor/e/r和Clean/Flip风险R，不干预上传。
- **B1**：同一训练和测量，唯一主动变化为历史E5 `rho=R×Q×D` 的三个关键层BDRC方向门控。使用当前原论文非归一化delta聚合；不是历史G5复现，不加入coverage scaling或Locked客户端Residual Gate。
- PRIMARY：固定5000样本validation上的Full Peak@300，保存完整七模型轨迹、Top5/10、final、分段均值和last30；完成后用Full验证峰值的同一global状态评价reserved半集。
- **不加入Checkpoint/恢复**；最佳模型仅在内存保留，不落盘。新启动入口不跑测试套件，也不自动重复短训；双进程只作nvidia-smi空闲显存查询。

```bash
pip install -r requirements.txt  # PyTorch/CUDA版本按服务器驱动安装
export CIFAR100_ROOT=/data/cifar100_raw
bash scripts/run_two_300_gpu1.sh    # 物理GPU1，同时运行B0/B1；至少6144MiB空闲
# 或显式串行（二选一，不要同时启动）：
bash scripts/run_serial_300_gpu1.sh
# 已有结果独立汇总：
bash scripts/analyze_performance_300.sh
```

本地只跑了B0/B1各2轮：Full `1.2→4.0%` / `1.2→3.4%`，B1干预非零但聚合相对变化仅约0.0249%；**没有性能提升证据，未启动正式300轮**。训练目录默认`results/performance_300/{b0,b1}`，分析输出`comparison`。输出已存在则拒绝覆盖；没有恢复命令。

参见 [升级源码说明](docs/300_round_upgrade_audit.md)、[固定实验协议](docs/performance_300_protocol.md)、[真实本地记录](docs/performance_300_local_execution.md)。验证/保留测试来自已历史暴露的官方test划分，不能与历史36.45%完整test峰值作严格直接比较。

## 历史50轮语义诊断（以下保留原说明）

独立的CIFAR-100语义证据实验项目。只有两个50-round版本：

- `ORIGINAL_SEMANTIC_EVIDENCE`：冻结共享投影＋私有adapter/classifier，按核实的早期实现联合优化task CE＋semantic CE；类别原型对旧Anchor评分。
- `FIXED_CLASS_SEMANTIC_EVIDENCE`：**训练完全相同，只把主要诊断评分参考改成固定类别方向**。100个类别使用128维空间的前100个标准基向量。不增加层、参数或训练损失，不声称表示被训练对齐到这些方向。

两个版本都记录Anchor score与fixed score，优先在同一批观测上比较。没有Gate、coverage scaling、BDRC、数据增强、KD或语义筛选。两组都使用论文非归一化delta聚合、gamma=1；不是AdaptiveFL逐坐标重新归一化聚合。

## Linux服务器

```bash
conda create -n semcons-probe python=3.11 -y
conda activate semcons-probe
# 先按服务器CUDA驱动安装相匹配的PyTorch/torchvision；requirements不绑定某个CUDA版本。
pip install -r requirements.txt
export CIFAR100_ROOT=/path/to/cifar100_raw
# 上述目录应直接包含 cifar-100-python/train 和 test；不自动下载或重分数据。
bash scripts/check_environment.sh
bash scripts/check_gpu1.sh
bash scripts/run_two_gpu1.sh
bash scripts/analyze_results.sh
```

所有命令从本项目出发，不依赖原Windows项目。`run_two_gpu1.sh`先在物理GPU1做两个短程、原本地预算的显存测量，再判断是否并行启动两个50轮。资源不足输出`GPU1_PARALLEL_NOT_SAFE`并退出，**不自动改串行、不改batch、不终止其他任务**。显式串行使用`bash scripts/run_serial_gpu1.sh`。

GPU映射为shell设置`CUDA_VISIBLE_DEVICES=1`，Python使用逻辑`cuda:0`。本地只有物理GPU0，已有本地验证明确使用GPU0；服务器GPU1检查必须在服务器实际执行，未伪造GPU1验证。

输出已存在即报错；需要重跑时主动指定新目录。独立脚本支持：

```bash
bash scripts/run_original_gpu1.sh results/original_run2
bash scripts/run_fixed_gpu1.sh results/fixed_run2
bash scripts/analyze_results.sh results/original_run2 results/fixed_run2 results/comparison_run2
```

## 本地已完成

最终两组各真实2轮，100客户端固定训练split，每轮10 slots、local_ep5、batch50；第2轮每组10次LOO，诊断验证500张，仅短程执行验证。原始与固定版本Full验证准确率轨迹均为1.2%、4.0%，不能据此评价方法优劣。两组初始化、dispatch及最终global参数相同；统计分类均为`INSUFFICIENT_EVIDENCE`。见`docs/local_validation.md`。本机没有运行50轮。

## 必须阅读的科学定义

- `docs/source_audit.md`：论文、原实现与当前版本的差异；无冗余哈希体系。
- `docs/method_definition.md`：联合语义分支、原型、Anchor、固定参考、LOO与round0。
- `docs/experiment_protocol.md`：固定配置、数据隔离、诊断轮次、分组与预测规则。
- `docs/gpu_memory_audit.md`：CPU/GPU驻留和并行边界。
- `docs/server_deployment.md`：服务器运行与失败排查。

`src/models/resnet.py`直接来自指定干净commit619c02e。`docs/original_sources/`保留核对的基线文件与早期SemConsFL源码文本，仅作来源材料，不作为隐藏依赖。代码只需本目录与外部CIFAR100图像缓存。

固定类别版本只检验**参考替换**。若未来要检验“固定类别监督能否主动对齐表示”，那会改变训练目标，是另一个需单独定义的实验，本项目不实现。
