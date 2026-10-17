# 实际执行的方法定义

## A与B共用的训练和聚合

客户端槽位从实际下发参数训练。原生任务输出保留，同时用最终stage表示经pooling、私有线性adapter、共享冻结QR投影、单位化和私有分类器构造semantic CE。loss=CE(task)+CE(semantic)，梯度共同进入主干；两个私有模块跨参与保存，optimizer每槽位重建。没有detach、增强、KD、coverage scaling、固定编码损失或验证标签监督。

同一物理客户端不同feature维度独立adapter，共享semantic classifier。同轮重复槽位按编号顺序更新私有bank，依据初始本地SemConsFL实现；task backbone仍从对应当轮dispatch开始。语义记录为每个真实occurrence，历史按每物理client每轮一次。

对类c，p_c=Normalize(mean_x Normalize(h(a(phi(x)))))，样本集是该客户端原训练索引去重、每样本一次；n_c>=5且平均向量非零才有效。所有样本类别计数和有效mask保留。

原Anchor分数 e=sum_c n_c*cos(p_c,mu_old_c)/sum_comparable n_c。先用M_t评分，聚合后按unique-client coordinate median与EMA .9更新M_(t+1)。历史W=5通信轮，r=clip(mean(e)-sqrt(sample_variance(e)/n),0,1)，不足2条时公式r=0但状态INSUFFICIENT_HISTORY。所有这些量均不影响client选择、训练或上传权重。

聚合：delta_s=local_s-dispatch_s；G=sum_s(n_s/sum_u n_u) Align(delta_s)；theta_new=theta+G。分母为所有当前槽位样本总量，无按坐标重归一化。未覆盖坐标更新为0。每个LOO重新去掉对应occurrence并重算sum n、覆盖和保护公式。运行gamma全部1，gamma_bar=1，保护从不触发。

首轮七档独立初始化，按实际dispatch取delta，round0只作训练/Anchor初始化，不用于正式LOO。后续统一从Full切片。初始化CPU构建，不保留七套GPU模型。

## 唯一候选B

q_c为R^128中第c个标准基向量，c=0...99。固定方向不依赖训练/验证样本、客户端标签分布或未来Anchor。fixed score=sum_valid n_c*cos(p_c,q_c)/sum_valid n_c。

A/B的训练、初始化、私有模块、loss、Anchor旁路和聚合相同。只有主诊断评分reference不同；每组仍同时记录两种score，可以基于同一批原型和utility严格配对比较。在当前确定性执行下，A/B backbone trajectories应相同；这不是错误，而是observation-only reference替换的预期。

固定方向不为表示提供训练约束，私有adapter/classifier仍存在坐标自由度。因此本实验可以检验“固定类别参考评分是否更有信息”，不能直接回答“固定类别监督训练能否对齐空间”。不新增第三种编码或新的训练loss。

## 实际效用

U_s=CE_val(theta_without_occurrence_s)-CE_val(theta_all)。正数表示本轮包含该更新有利，负数表示该次边际效果不利。重复client只排除对应slot，保留它的其他occurrence；这不是对该client所有更新的删除。

简单方向特征为该slot原生delta与**排除该slot后的实际非归一化聚合方向**的cos，只在共同实际覆盖坐标计算，不使用验证梯度/utility生成方向。不同原生宽度不直接裁Full验证梯度来冒充native梯度。

验证模型与批次顺序固定，BN track_running_stats=False。完整验证集按固定混合顺序、batch128、保留最后一批计算CE总和/N；所有反事实都用同一规则。
