# 300轮升级：与科学含义直接相关的源码核对

本轮只增量扩展已有项目。遵从用户最新要求：不新增测试体系、不跑旧测试套件、不加入Checkpoint或恢复，不在本地启动正式300轮。旧50轮入口、Original/Fixed-Class分析和历史结果保留。

## 原有项目实际执行内容

1. Original和Fixed-Class共用 `src/semantic.py:train_and_measure`。主干、私有adapter与私有classifier联合优化 `CE(task)+CE(semantic)`；共享QR测量投影冻结但传播梯度。Fixed-Class只改变评分reference，没有固定类别编码监督。两组本来应产生同一训练轨迹。
2. `src/core.py:aggregate` 已是原论文的非归一化delta聚合。`delta=local-dispatched`，权重是每slot样本数/全部slot样本数，按prefix累加；不存在逐坐标覆盖分母。未覆盖坐标保留旧值。旧入口gamma=1，Gate完全关闭。
3. 已有旧Anchor、原型一致性e、W5历史r、occurrence LOO；原先没有Clean/Flip的S/R，也没有BDRC。旧LOO只移除一个slot；本轮增加删除同一物理客户端全部slot的grouped utility。
4. 现有七个独立初始模型在CPU构建，保留原入口额外构建未使用Full的随机数消耗；round0从各自真实dispatch取差，round1起从Full切片。新B0/B1沿用同一初始化函数、300轮私有dispatch计划与每slot数据顺序。

## 历史实现确实可获得

可直接读取本工作区的以下实际源码，而非依据文件名推断：

| 来源 | 已确认机制 | 本轮处理 |
|---|---|---|
| `experiments/semconsfl_v2_sweep.py:StabilityObserver/SweepServer` | E5：clean/flip类别原型cos映射为S，R=1−S；unique-client ALL_REF、Q、D、BDRC；无客户端标量Gate | 只迁移这一套有界方向干预公式 |
| `experiments/semconsfl_v2.py:PassiveSemanticLocal` | 历史E5语义分支使用detached主干特征，私有CE不回传主干 | 与当前项目joint CE不同，不伪称历史训练轨迹复现 |
| `experiments/semconsfl_v2_coverage.py:MATRIX/CoverageSession` | G5=E5+sqrt(7/c)任务梯度缩放 | 本轮禁止coverage scaling，未迁移 |
| `experiments/semconsfl_locked.py:LockedServer.aggregate` | 新Locked有支持数约束、H、客户端gamma、总体保护更新 | 没有选择这套Gate，也没有混入B1 |

历史E5/G5使用原AdaptiveFL聚合，与本项目已采用的原论文非归一化聚合不同。B0/B1保持当前聚合不变；本轮实验名称是 **E5_ALL_REF_BDRC_TRANSFER**，不是E5/G5的同协议性能复现。本次公开更新不包含新增的本地历史源码摘录。

## B0与B1的唯一主动差别

- B0 `SEMANTIC_MEASUREMENT_ONLY`：联合CE、旧Anchor/e/r和Clean/Flip测量全部照常，原始delta直接进入当前非归一化聚合。
- B1 `VERIFIED_SEMANTIC_BOUNDED_GATE`：训练、测量、调度和聚合底座相同，只在三个关键卷积层对delta应用历史E5的BDRC方向门控。
- 这里的Gate系数是方向残差保留率 `1−rho`，不是原论文beta筛选、Locked客户端gamma或整客户端标量乘法。总体gamma_bar仍是1，保护更新不触发。
- 历史缺失仅记录为历史不足；r不作为本轮门控输入。缺失S/无参考/不可计算cos时rho=0，更新保持原样。

e=sum类样本数×cos(当前clean原型,旧Anchor)/可比类样本总数，越高表示历史一致。

S=sum类样本数×(1+cos(clean原型,flip原型))/2 /有效类样本总数；R=1−S，越高表示变换不稳定。e与R的方向相反，不互相替代。固定类别评分仍同步记录。

Clean与Flip使用同一已训练adapter、冻结head、相同去重训练样本、相同batch顺序，完整native模型eval/no_grad；flip不进入CE训练。Anchor只接收clean原型。当前joint CE空间下可合法计算R，但历史detached空间的数值分布和效益不能直接沿用。

## 300轮资源、日志及科学边界

- 新入口 `src/performance.py`：每轮7profile验证；10个预定轮次做occurrence/grouped LOO及四种只读决策对照；只保存JSON/CSV和stdout日志。
- 最佳Full验证模型只保留在RAM，完成300轮后对同一个global状态的7profiles评价保留测试集；不保存模型Checkpoint、不提供恢复。中途失败保留已完成标量日志，从头重跑必须使用新输出目录。
- 所有非活跃模型、私有bank和上传delta在CPU。GPU一次只放一个local模型或一个评估模型；不把十个反事实模型并行放GPU。
- 诊断使用完整5000验证样本，成本高于普通轮次；本地短执行只使用其前500个固定混合样本。因此本地计时不能直接作为完整300轮精确工期。
- 当前没有能核验36.45%全部配方与评价口径的同协议产物链。它保留为用户提供的历史描述，不当作B0，也不与新验证半集的peak直接相减。历史官方完整test若曾用于选峰，新reserved半集也不是从未被看过的独立测试集。
- 不要求运行任何旧测试套件、文件hash框架、GPU1重复短训或恢复检查。实际短执行与一个固定向量算例见 `performance_300_local_execution.md`。
- 局部数值变化不等于有效机制，低风险仍可能使300轮干预很弱。不会为保证“Gate活跃”而放大R、Q、D或修改强度；按真实日志报告弱干预和负面结果。

本轮不改变任何历史G5/E5/Locked代码、原始模型、分区或Locked Method。只扩展本独立项目。
