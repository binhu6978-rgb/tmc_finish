# 本地短执行：300轮入口升级

状态：**READY_FOR_300（代码及启动配置已准备，正式训练尚未开始）**。

本地物理GPU0：RTX4080SUPER16GB，PyTorch2.7.1+cu128。服务器物理GPU1没有在本机实测。本次不跑旧测试套件、不新增测试脚本或hash框架；按用户要求只执行B0/B1各2个真实通信轮，每个slot仍为5个local epochs、batch50，不用假数据缩短本地训练。固定验证500个样本，仅作低成本执行观察；正式配置是完整5000validation。

| 真实记录 | B0 | B1 |
|---|---:|---:|
| 完成轮数 | 2 | 2 |
| 完成训练slot | 20 | 20 |
| Full验证round1 | 1.2% | 1.2% |
| Full验证round2 | 4.0% | 3.4% |
| 总wall | 36.547s | 38.378s |
| CUDA allocated峰值 | 526.904MiB | 526.904MiB |
| CUDA reserved峰值 | 754MiB | 754MiB |
| 真实修正slot-layer | 0/60 | 60/60 |
| 平均rho | 0 | 0.0008637241 |
| 平均方向保留率 | 1 | 0.9991362759 |
| 最低方向保留率 | 1 | 0.9985543184 |
| 聚合norm ratio均值 | 1 | 0.9999406153 |
| 聚合相对变化均值 | 0 | 0.0002492415 |
| 历史r有效unique-client比例 | 0 | 0 |

两组300轮预生成dispatch计划相同。每组round1（0based）实际完成10个occurrence LOO、9个physical-client grouped LOO及4个固定上下文决策对照。每组20条slot证据、18条unique-client/round证据。没有访问reserved测试集，没有任何模型Checkpoint。

**实际变化虽非零，但幅度很弱**：B1聚合相对变化约0.0249%，平均R约0.00413。这不能包装成有优化意义的“强干预”。两轮观察没有显示性能提升，不能预判300轮表现，也不为此放大公式。W5在两轮内没有有效历史是预期事实，r不是本轮B1输入。

## 同一B0原始更新上下文，round0=1

| 决策 | validation CE | delta CE vs No-Gate |
|---|---:|---:|
| No-Gate | 4.5417708740 | 0 |
| Real Semantic Gate | 4.5417805176 | +0.0000096436 |
| Profile/Round Shuffled Gate | 4.5417833252 | +0.0000124512 |
| Aggregate-norm-matched Nonsemantic | 4.5417818604 | +0.0000109863 |

该轮四者accuracy均4.0%。真实Gate的CE没有改善；几微量CE差距不能作为证实语义决策价值的证据。B1自己分叉轨迹上的相同诊断也保留在原始文件，不与B0直接混合。

## 一次数值算例，不是新测试体系

固定三个更新 `[1,0]、[1,0]、[-1,0]`，参考中位数 `[1,0]`；两正向客户端R=0，负向客户端R=1。实际函数输出rho `[0,0,1]`，更新变为 `[1,0]、[1,0]、[0,0]`。等样本非归一化聚合从 `[1/3,0]` 变成 `[2/3,0]`。说明函数能保留低风险更新并去掉指定高风险负向分量；不是论文性能证据，也未用于调整强度。

## 位置与命令

完整原始运行保留在外层工作区：

`experiments/semantic_utility_probe_performance300_local_20261017/{b0,b1,comparison}`。

本次公开代码更新不包含新增的本地原始标量记录、环境元数据或模型权重。上述数字为本地短执行的摘要。

```bash
CUDA_VISIBLE_DEVICES=0 python -u -B -m src.performance --arm b0 --rounds 2 --local-check --validation-limit 500 --data-root /path/to/cifar100_raw --output /new/local/b0
CUDA_VISIBLE_DEVICES=0 python -u -B -m src.performance --arm b1 --rounds 2 --local-check --validation-limit 500 --data-root /path/to/cifar100_raw --output /new/local/b1
python -B -m src.performance_analysis --b0 /new/local/b0 --b1 /new/local/b1 --output /new/local/comparison
```

本地B0运行结束后补充了每轮seven-profile均值日志与聚合finite报错保护；均未改变数值更新。已有B0标量日志保留原样，其七模型均值可从7条validation直接计算。未重复短训来补日志。

READY仅表示可以执行已定义的300轮研究性比较；不代表预测B1优于B0，也不代表已证明历史E5/G5迁移有效。服务器仍需入口的轻量空闲显存查询。中断只能从头新目录重跑，无恢复命令。
