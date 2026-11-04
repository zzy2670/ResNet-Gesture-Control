# B协议固定实验规则

本轮从人员互斥划分检验新人员泛化。全部同一user_id照片必须同组；精确重复内容如跨用户需按连通人员组共同分组，人员、照片ID、内容哈希跨组为零。固定seed42，目标训练／验证／测试人数约8∶1∶1，实际人数、照片数、手数及原18类／合并6类分布以独立分组审查为准。不得按测试成绩重划。

## 与A保持一致的算法

- 六类标签：1 peace/peace_inverted，2 stop/stop_inverted/palm，3 fist，4 like，5 dislike，其余/no_gesture为0。
- 完整照片检测→1.2倍手框长边正方形裁图→越界补黑→224双线性→ImageNet归一化→ResNet六类→固定阈值与多手汇总；停止3优先，0不抵消另一有效单动作，冲突0。
- 三分类器各从官方ImageNet V1初始化，不使用A微调模型。B检测器从官方ImageNet MobileNetV3初始化，仅B train人员的背景／手监督训练，三个分类器共享该B检测器。
- 每架构24个TPE试验，仅冻结ImageNet特征上的Dropout+Linear头，每试验18轮。seed42，startup4；AdamW/SGD，学习率1e-4—3e-2(log)，weight decay1e-6—1e-2(log)，dropout0—0.4，label smoothing0—0.12，训练类频率平方根倒数权重开关。以完整val macro-F1选取。
- 分类微调仅layer4和fc，AdamW，1e-4/5e-4初始学习率，cosine上限18，patience4，batch128，AMP、clip5。真实完整轮数如实保留，不保证满18轮才算结束。
- SSDLite手检测16轮，batch24，SGD lr0.003/momentum0.9/wd5e-4，cosine、AMP、clip10；按val score0.5/IoU0.5的F1选权重，不仅挑recall。保持训练时完整MobileNet尾部，strict加载。
- 冻结模型后完整val只推理一遍，检测阈值0.3/0.5/0.7×分类阈值0.3/0.5/0.7/0.9共12候选，以macro-F1、停止误运动、0误运动及更高阈值确定唯一配置；冻结权重哈希后完整test。不得依据B test继续改权重、阈值或分母。

## 数据与执行边界

原A源代码及产物不改。B新输出只在本目录和data/B_split_verified_20261102。复用train_A.py和compare_resnets_A.py函数，在本进程绑定B路径和架构；Optuna研究名改B，算法不改。所有官方初始化文件哈希须命中其官方文件名前缀；不允许联网下载或用A权重冒充初始化。

controller每阶段独立日志、PID与退出，成功凭据记录源码/正式产物哈希；恢复只跳过可核验成功，失败须复盘后显式retry，不盲重跑或覆盖已测试结果。单GPU串行，CPU独立准备可并行，不关机、不重启、不运行旧日期A预算。

人工框与完整照片成绩分别报告，完整逐样本预测/每类混淆/全部错误方向/停止→0和误运动均保留。CPU整图固定前100test、10预热、8线程batch1；纯分类同100预处理手图20预热3×100，float32，模型加载不计。测速不用于选模型。最终独审与A/B报告另行完成，不把执行exit0当整个目标验收。
