# B0 / B1：固定300轮Full验证峰值实验

## 固定配方

CIFAR-100，原ResNet18嵌套族，Dirichlet alpha=.3，iid=0，seed=1，100clients，每轮10个occurrences；七档(.4,2/3/4)、(.66,2/3/4)、(1,2)，available资格与4:3:3能力分区，有放回重复slot。原cached split不重生成，50000训练样本不变。local_ep5、batch50、dropLast，SGD lr=.01*.998^round、momentum=.5、wd=.0001；BN不跟踪running stats，无训练增强。

两组使用原项目联合task CE+semantic CE、本地私有bank、冻结共享QR投影、原Anchor/W5和相同评估。新300轮配置仅覆盖轮数、共同的测量Flip、固定诊断轮次及B1开关，旧50轮配置仍在。

## 唯一B1干预：经源码核实的E5方向Gate

1. 各类clean/flip有效原型的稳定性S_c=(1+cos(p_clean,p_flip))/2。样本数加权得到S和R=1−S。重复物理client同类先按count平均S_c，类权重取最大实际count，与历史E5一致。
2. 三个关键卷积层按七profile输出宽度max/min降序、Full参数量降序、state顺序选取，不使用验证结果。实际层名写入config.json。
3. 每物理client在每层选实际覆盖最多的一个slot投票，覆盖相同取first。ALL_REF每坐标只取实际覆盖者中位数，偶数取中间两数平均；参考包含自身。没有覆盖的坐标不改。
4. u=cos(delta,reference)，只在共同有效覆盖坐标；零向量不可计算。Q=clip((median(unique u)−MAD(unique u)+1)/2,0,1)。与历史E5一样不额外加支持数阈值。
5. D=(1−u)/2，rho=R×Q×D；无信号/无方向/不可计算时rho=0。reference平方范数<=1e-12保持原向量。
6. p_plus=max(0,<x,g>/(||g||²+1e-12))g；x_new=(1−rho)x+rho p_plus。仅修改这三个卷积层共同覆盖坐标，其余参数原样。
7. rho∈[0,1]；不增加额外强度，不搜索阈值。它可以阻尼正余弦更新中的正交分量，因此负cos少不代表绝对没有干预。

保留率1−rho不是全客户端gamma。实际delta聚合在B0/B1都为：

`theta_next=theta+sum_s n_s/sum_u(n_u) * Align(delta_s_or_corrected_s)`。

分母含所有当前occurrences，无按坐标重新归一化；duplicate仍独立计样本权重，未覆盖坐标无更新。gamma_bar=1，eta=1，无额外服务器缩放、无coverage scaling或Locked Residual Gate。

## 验证与保留测试

沿用现有 `evaluation_split(seed=20261017)`：CIFAR100官方test每类固定50个作validation、50个reserved，混合顺序固定、batch128、不丢最后一批。与训练集不重叠，但官方test历史上已暴露，因此本轮是探索性划分，不能称独立正式盲测。BN使用当前batch统计，所有比较保持相同分批和顺序。

PRIMARY为Full Validation Peak@round0=0…299，first max为best round，同时报告0/1based。完整每轮每profile accuracy/CE都保存。Top5/Top10是最高5/10个通信轮accuracy的均值。

辅助记录Full final、0–99/100–199/200–299分段均值、last30、trajectory、七profiles各自peak/final/last30和七profile平均accuracy。各profile自身peak仅诊断；最终reserved主表使用Full验证峰值所对应**同一global模型**的七个切片，不平均来自不同轮次的profile peaks。CRD未在本项目定义，不临时引入。

不落盘Checkpoint或RNG/optimizer恢复状态。最佳global模型仅RAM保留；300完成后reserved评价一次并输出 `selected_validation_test.json`，含所选轮次及七profile结果。中断后模型不可恢复，仅保留完成轮次统计，不声称已得到完整Peak@300。

## 固定语义效用与决策诊断

预定0based轮次 `[5,19,49,74,99,124,149,199,249,299]`，不是看accuracy选快照。前100轮Early、中100轮Middle、后100轮Late。

同一轮原始delta、同一个base和固定validation：

- occurrence LOO：删一个slot，重新计算总样本分母；U=CE_without−CE_all。
- grouped LOO：删同一物理client的所有slot并重算分母。物理client每轮只留一条merged证据，用对应grouped utility分析。
- no Gate、真实R Gate、round/profile分层shuffled R Gate、与真实Gate最终聚合范数匹配的非语义缩放控制。正delta CE表示比raw更差。
- shuffled只在当前同profile的valid unique client间重排，重复slot共享对应风险；跨profile单独参与；不同值时identity排列作一次固定rotation，缺失保持中性。Q/reference仍由相同原始更新和实际物理client计算。
- 非语义控制用s=||G_real||/||G_raw||乘raw聚合；受向量抵消影响s可能>1，不截断、不声称匹配各slot修正剂量。
- B0诊断为主要证据关联，B1单独保存为分叉轨迹的辅助分析。R和−R同时报告，避免把高风险误作高效用。

所有反事实都只读，不写回训练模型、不进入Gate输入。按原始e、R、r、update norm/cos、样本数、类覆盖，输出总体、profile组、阶段和within-round ranking；不做IID样本p值。没有保存local向量供事后重放；这些离线性质计算在当轮材料可用时完成，只持久化标量结果。

## 实际作用与弱干预

每轮保存所有slot/layer的R/Q/D/rho、保留率、修正范数和比例、负向残余、unique reference/覆盖、实际聚合cos与norm ratio、Anchor类别数、历史有效率。

报告标记（只作描述，绝不反馈训练）：active slot<1%，或所有轮聚合相对变化<1e-6 => GATE_INACTIVE。超过半数轮norm ratio<.8，或norm ratio<.95且cos>.9999 => GLOBAL_ATTENUATION_RISK。否则只写方向干预实际发生；不能据此声称强度充分或有优化收益，仍须展示真实幅度。

用户要求避免全程无作用，但没有公式能在不改算法的前提下预先保证这一点。如果真实R持续很小，照实报告弱干预，不调参救结果。

## 服务器运行

```bash
export CIFAR100_ROOT=/data/cifar100_raw
# 目录内有cifar-100-python/{train,test,meta}
# 安装与服务器驱动匹配的PyTorch和requirements，然后二选一：
bash scripts/run_two_300_gpu1.sh
# 或显式串行：
bash scripts/run_serial_300_gpu1.sh
# 单独启动：
bash scripts/run_b0_gpu1.sh
bash scripts/run_b1_gpu1.sh
# 单独汇总：
bash scripts/analyze_performance_300.sh
```

物理GPU1映射为logical cuda:0。并行入口只查询nvidia-smi当前空闲显存，保守要求至少6144MiB，不再自动跑短训或测试。当前本地两组reserved峰值均754MiB，不包含CUDA进程上下文；6GiB要求包含额外余量，仍是跨硬件估计。显存不足报错，显式串行，不改batch/model/数据配方，不杀其他任务。串行不经过双进程门槛。

输出分别为 `results/performance_300/b0`、`b1`、`comparison`。训练目录存在则拒绝覆盖；重跑用单组入口显式指定新目录。stdout记录于logs。未自动启动正式训练、未承诺服务器GPU1已实测；无恢复命令。
