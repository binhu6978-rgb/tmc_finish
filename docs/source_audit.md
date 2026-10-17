# 与实验直接相关的来源核对

本次遵从用户最新要求：保留公式与数据定义的必要检查，不构建逐文件哈希、复杂冒烟或软件工程审计体系。10项小型数值检查和两组各2轮真实执行服务于论文实验正确性。

## 来源

- 原论文：`SemConsFL: Conflict-Aware Aggregation for Heterogeneous Mobile and Edge Federated Learning`，用户本地SemConsFL.pdf，SectionIII Eq7–27。
- 干净AdaptiveFL： https://github.com/binhu6978-rgb/AdaptiveFL-support-aggregation/tree/619c02e 。核对Training_AdaptiveFL.py、Fed.py、Update.py、resnet.py、get_datatset_new.py、options.py及原入口。必要源码文本保存在`original_sources/`。
- 最早本地SemConsFL：当前历史commit `31c8e8cf73ea02f509dcf442e91d792c4542b570` 中的`models/SemConsFL.py`，存为`original_sources/models/SemConsFL_initial.py.txt`。其训练明确是未detach的task CE＋semantic CE，adapter与主干共同优化。当前工作树同文件已改为detach；**本项目未直接导入工作树版本**。
- 中文Locked Method及G5报告用于区分后续改动，没有继承coverage、BDRC、Flip、Residual Gate等算法。

| 原定义/来源 | 数学含义 | 本项目位置 | 使用与差异 |
|---|---|---|---|
| 原论文Eq7；初始实现FrozenMeasurementHead、ClientSemanticBank | z=h(a(phi(x)))；共享冻结h、私有a | `src/semantic.py:modules` | 同结构；d0=512、d=128来自已核实本地实现；QR投影、零补齐identity adapter、零初始化私有classifier也是源码解释，不声称论文唯一指定 |
| 原论文III-B2联合训练文字；初始LocalUpdateSemConsFL.train | task CE＋semantic CE；语义梯度经过adapter回主干 | `src/semantic.py:train_and_measure` | 忠实采用已确认初始代码的等权两CE；论文未给出唯一的两损失权重公式，明确标为源码的具体实现解释 |
| 原论文Eq8–9 | 单样本embedding归一化后按类平均、再归一化；n>=5 | `train_and_measure`测量段 | 同定义，完整native模型eval；不翻转、不加入prototype/Anchor损失 |
| Eq10–12 | 按不同客户端做类别coordinate median、单位化、EMA .9 | `Memory.update` | 同公式；偶数取中间两者均值；重复物理client先合并，一轮一票，是原文未细述重复调度的实现约定 |
| Eq13–14 | 与实际可获得旧Anchor比较，只在共同有效类别按n归一化 | `Memory.score` | 同公式；未初始化类别记缺失，不填0冒充有效证据 |
| Eq15–17 | 最近5通信轮均值减标准误，少于2条r=0 | `Memory.reliability` | 记录公式r=0和可用性分开；主日志r=null、INSUFFICIENT_HISTORY，不把历史不足写成不可信；无干预 |
| Eq18–24 | β、可信子集和方向Gate | 无运行路径 | 本阶段明确禁用；不计算后再偷偷应用 |
| Eq26 | theta_new=theta+eta sum(n_s/sum n)*Align(delta_s) | `src/core.py:aggregate` | gamma固定1；所有槽位包括重复独立计权；**无逐坐标覆盖分母** |
| Eq27 | gamma_bar不足delta0时补总体系数，0门控不更新 | `aggregate` | 数值函数保留公式，运行gamma_bar=1，所以保护不激活；没有保留一套隐藏语义gate |
| Eq2及覆盖说明 | 未覆盖坐标不进入和、无覆盖保留旧值 | `slices`、`substate`、`aggregate` | 原生prefix；数值检查覆盖不相等及无覆盖坐标 |
| 干净Update.py | 5ep、batch50、dropLast、SGD .01*.998^t/.5/1e-4 | `train_and_measure` | 任务配方相同，但联合语义CE额外改变主干优化，不能称为纯AdaptiveFL observer |
| 干净Fed.py、Training_AdaptiveFL.py | np.choice7有放回；available client资格 | `src/core.py:dispatch` | 同采样分布及资格；独立RandomState/Python RNG保存共享plan，明确改变了与旧全局训练RNG交错的协议，不宣称旧轨迹逐bit相同 |
| 干净resnet.py | 原架构、7profiles、BN无running stats | `src/models/resnet.py`、`make_model` | 直接复用commit源文件，没有改网络容量；全部非活跃模型CPU驻留 |

原main还先构建一份未使用Full模型，再进入七模型构建；`build_initial`保留这次初始化随机数消耗。最先的两轮本地开发检查曾省略该未使用对象，最终确认后在新目录重新执行两组短程检查；未改训练公式，未运行正式50轮，不覆盖初次记录。

## 两个需要明确的边界

1. 原论文描述的每轮共同global切片与原AdaptiveFL首轮七模型独立初始化有差异。依本任务要求保留七独立初始化，delta从各自实际dispatch相减，round0仍使用同一非归一化delta函数；因此round0是明确的初始化协议例外，不做正式utility。round1起所有profile都从上一轮Full切片。
2. B的100个固定方向取128维标准基前100行，仅改变评分参考，不改变联合CE、classifier或参数量。它是新的reference-only candidate，不冒称原论文方法；也不宣称这些方向参与训练或建立了类别表示对齐。

没有继承Support、Temporal、Full-Access、Quota、Bridge、旧KD、多视图或任何服务器抑制。用户附件中的这些历史结果仅作问题背景。
