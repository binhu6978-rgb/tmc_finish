# 冻结的两组科研协议

正式每组50轮，seed1，CIFAR100，Dirichlet .3固定缓存，不重新生成。100clients，每轮10slots，profiles(.4,2/3/4),(.66,2/3/4),(1,2)，4:3:3 eligibility，available、有放回。local_ep5、bs50、drop_last=True、SGD .01*.998^round、momentum .5、wd1e-4，无augmentation。

`data/splits/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json`是现有文件的直接拷贝。读取时检查min262/median491.5/max757/sum50000，索引恰为0...49999且不重复；文件名含bal不代表其实际算法是balanced。无重新分割路径。

官方test每类100张，以独立seed20261017分成50+50，再固定打乱类间顺序，得到5000 diagnostic validation、5000 reserved。indices和每类数写入evaluation_split.json。训练50000不变；保留集不用于本阶段评价。历史使用过官方完整test，所以reserved不声称绝对独立未见。

LOO固定round0-based [5,10,20,30,40,49]，每轮十个occurrence，预期60条utility/组。每轮共11个Full CE评估，不为寻找好结果改变轮次。local-check可以缩为1-3轮并减少验证样本，输出明确标短程；它不改变local_ep、batch或模型。

dispatch用独立np.RandomState(1).choice7与random.Random(1).randint，根据原资格生成50轮共享计划，保存在各组dispatch_plan.json。loader采样使用seed*1000003+round*1009+slot*97的私有generator；额外测量/evaluation也有独立loader generator。初始7backbones CPU按固定顺序生成。同seed不被当成全轨迹相同的证明；分析实际比对每轮profiles和clients，要求mismatch0。

## 离线预测规则

- 主分数越大预期U越正；Spearman、Pearson与AUROC报告原方向，不根据结果翻转符号。
- 1000次bootstrap按整个round有放回抽样；少于3个round不生成区间，6个正式聚类仍很少，CI标探索性，无IID p值。
- 增量预测：固定ridge penalty1，用log1p(sample_count)、log1p(update_norm)、排除当前slot的cos、width/depth和profile one-hot预测连续U，逐round留一预测；加入semantic score对比MSE、AUROC、balanced accuracy，所有标准化只在训练round拟合。
- 增量分析至少40有效obs、6round且正负各至少5；不足标INSUFFICIENT_SAMPLES。语义null不填假0，不足历史保留状态。
- 同原型共享评分，另报告只在原Anchor与fixed reference均有效时的公平比较；固定参考冷启动更早可用不自动视作预测优势。
- 能力Small/Medium/Full；阶段Early0–15/Middle16–32/Late33–49；有效类别数<25、25–49、>=50。小组不足20条或3round标INSUFFICIENT_SAMPLES，不强推子组结论。
- Anchor-free补充为原型归一化前浓度；全零中性向量余弦无定义，不伪造无Anchor的分数。

## 保守的结论标签

正式完整50轮、有40有效obs及6个round后才考虑USEFUL：正Spearman的round-bootstrap下界>0、AUROC>=.6且加入语义使round-held-out MSE降低同时满足。正相关且AUROC>.5但未达全部条件为WEAK；负相关CI上界<0且AUROC<.5为MISLEADING；其余INSUFFICIENT_EVIDENCE。单seed与少量round的标签是探索投入判断，不称统计显著或普遍因果。高相似度均值>.9却未达USEFUL时额外标HIGH_SIMILARITY_BUT_LOW_UTILITY_PREDICTIVENESS。

本阶段不寻找更好Gate、EMA、编码或loss。两个50轮结束后分析，停止；没有自动150/500/1000、多seed或第三种方法。
