# 本地真实运行与交付状态

日期2026-10-17。只做少量执行检查，没有正式50轮。没有逐文件哈希体系、训练参数搜索或复杂工程gate。

## 最终短程

物理GPU0，RTX4080SUPER，torch2.7.1+cu128；CIFAR100使用现有图像和指定固定split。A/B各2通信轮，每轮10slot、每客户端原5本地epoch、batch50，实际样本不缩减。诊断验证从固定5000序列取前500，保留子集不访问；round0排除utility，round1每组10次完整leave-one-occurrence-out Full CE。

| 项目 | A Original | B Fixed-Class |
|---|---:|---:|
| 实际完整通信轮 | 2 | 2 |
| 实际客户端槽位 | 20 | 20 |
| Full diagnostic-validation accuracy | 1.2%,4.0% | 1.2%,4.0% |
| Full LOO utility记录 | 10 | 10 |
| wall秒 | 26.074 | 25.757 |
| peak allocated MiB | 527.84 | 527.84 |
| peak reserved MiB | 756 | 756 |
| GPU模型构建时allocated峰值MiB | 107.58 | 107.58 |
| Gate/保护实际触发 | 0/0 | 0/0 |
| 统计结论 | INSUFFICIENT_EVIDENCE | INSUFFICIENT_EVIDENCE |

初始7backbones逐tensor相同；实际profile_dispatch_mismatch=0、client_dispatch_mismatch=0；最终global逐tensor相同。记录了prototype有效mask、类别n、旧Anchor支持、冷启动、r历史不足状态、原/固定reference score、每个utility的全量与排除CE。分析完成，缺少足够round时CI/增量预测正常返回不足样本，没有把短程相关性当科研结论。

最初开发阶段另各做过2轮，随后发现原main在七模型前还有一份未使用Full初始化。最终入口保留该随机数消耗，在新目录重新完成上述两组；初次日志和权重另行保留，没有覆盖或利用准确率挑初始化。总计本地开发8个通信轮，正式50轮为0。末次仅增加只读训练/原型wall计时字段，不改优化；较早本地日志未分别计时的字段保持缺失，未补造开销。

## 必要检查

10/10小型检查通过：非归一化聚合/LOO分母/无覆盖坐标、保护公式边界、原缓存统计、7档初始化和形状、资格与重复调度、Anchor旧新时序/缺失历史、验证分层隔离、真实CE下utility符号、统计单类别/少样本边界，以及按round留出的增量预测和聚类bootstrap在受控数据上正常工作。受控数值不作为论文性能结果。Bash全部脚本语法检查和Python解析通过。

受控统计检查暴露本机混用NumPy/Torch数值库时小型BLAS求解停滞；仅离线ridge的小矩阵计算改为显式归约和带主元消元，仍求解同一个固定penalty1的正规方程。已停止确认归属的自有离线检查进程；没有终止训练或其他任务，也没有为此重跑GPU。

这不是原论文性能复现或50轮机制证据。早期联合loss来自已确认本地初始实现；对论文定义差异详见source_audit.md。

## 服务器状态

**PROJECT_READY_FOR_SERVER**：可移植代码、依赖、固定split、两组脚本、分析及必要真实执行检查均已完成。

**GPU1_PARALLEL_NOT_YET_MEASURED**：本机没有物理GPU1，没有伪造GPU1或双进程并行实测。实际服务器必须通过run_two_gpu1.sh内资源测量；不安全即GPU1_PARALLEL_NOT_SAFE退出，不自动串行。

简短原始摘要、日志、实际dispatch与分析记录保存在`docs/local_validation/`。本地大模型/快照另保存在原工作区experiments/semantic_utility_probe_local_20261017，避免上传约2GB无用权重；它不是服务器代码依赖。
