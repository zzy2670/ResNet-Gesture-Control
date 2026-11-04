# 当前任务状态

更新时间：2026-11-02 15:46 Asia/Singapore。此文件仅属于本聊天的A本机手势训练，不属于旧Week4远程任务。

## 当前授权

- 主任务A：本机RTX5060 Laptop8GB、TPE、16:50前训练结束，自主完整测试验证、Word实验报告。
- 若时间够，对比ResNet34/50的准确率与速度，大表格；同A名单和同手检测器。
- B仍不划分/不训练，不改prepare_datasets.py或三组manifest。
- 同时处理ResNet视频图文笔记。最初两段已完成并通过图文检查；第三链接存在4P范围问题，已用异步问题等待用户选择，不能把P1冒称全系列。

## 真实运行（下方历史阶段；以最新完成记录为准）

- 主流程已全部完成。原session20580在校准时因推理工厂尾宽不一致exit1；原训练18微调/16检测均自然exit0，不重训。修复后session81336从calibrate恢复，校准、完整test、CPUlatency均exit0。旧失败日志calibrate_first_failure_full_tail.log保留。
- 额外对比监督：当前exec session49665，run_A_comparisons.py。ResNet34已24TPE/18微调及完整test自然exit0；正在ResNet50微调，PID39260，15:41:32开始，目前3完整轮，每轮31095手图。GPU健康勿重启。旧68180只是被修复SHA替换的等待监督，不是GPU任务。
- 训练时train_A.py SHA93e03db74bf5218db2fc3bdf2553c605c2dded2d2b459cca46e8c42c904c6303；当前仅修推理工厂后的SHA8f63912166b2ecee0b31cacf984904e646cf43d2b722eb4a4167956a4b5e1b9d，两budgetrunner均pin新SHA。推理也以缓存ImageNetV1主干创建full-tail，再strict加载原训练checkpoint；不要传重复reduced_tail参数。
- compare_resnets_A.py冻结SHA3bd9e96999917edf07a1738b0ab6eb8710405848cc2baa1691303e97ec944a3f。
- 主训练硬截止：finetune15:05，detector16:25；校准16:35、test16:43、CPU16:48。额外34训练16:00、50训练16:33，均独立进程树截止监督。
- 根负责GPU调度和最终Word，不要重复启动训练。所有阶段console和events日志都在本目录。

## 数据与已完成成绩

- 固定A原图25468/3185/3180；派生手图31095/3872/3917，见data_audit.json。A同人可跨组，不能称人员独立泛化。
- TPE24 COMPLETE trial，每trial18轮冻结特征上的头部训练；best23 val macro-F1 .7852070609787477。不是全网络TPE。
- 18微调18完整轮，best15 val macro-F1 .9892619592644359，classifier_best.pt SHA611023f0954d8e756b4e6b4620ea288d9ae933903958f4c0e86887750942ec94。仍不是test。
- 检测已16完整轮，best10，SHA9f2a395f0b942b03d3d3ddc96e386728bd0c3681ade0fe2963954aab455b7f5e；val score.5/IoU.5 P.968152866/R.667355372。校准只val选det.3/class.3。
- ResNet18 test：手图3891/3917=99.3362%、macroF1.99102814；整图2977/3180=93.61635%、macroF1.90449150。0误运动16/1773、stop误运动1/173且stop→0有35；det score.3 P.8825933/R.8271636。CPU整图100张mean32.253655ms/P9541.801185ms。完整预测均存，禁止因分数再调参。
- ResNet34 test：手图99.18305%、F1.98836784；整图93.14465%、F1.89760790；阈值det.3/class.5。18完整微调轮、best15、845.448s。共享detector同权重，但det匹配TP3241比18的3240多1，不得硬复用18检测结果；细小数值路径差异待审者解释，不重测调参。
- preflight_independent_review.json已核全38884裁图记录、3manifest哈希、TPEsqlite与代码/截止。独审agent cable_paper_presentation_study等待最终逐样本预测独核。

## 剩余工作

1. 监控主/对比预算监督的真实PID、退出码与日志。不重跑已完成stage、不用test调参。
2. 主流程上述文件已保存；final_independent_review.json已完整核3917/3180所有key、概率、聚合规则、IoU与24张CAM公式/矩阵像素。34独审在进行。a_report_author是6.1-sol/high正文solewriter，仅写实验报告正文.md/coverage；1–6章已审，34大表已加，等50+统一速度后续。root排Word与图表。
3. 对比完成后运行benchmark_A_classifiers.py，三架构同100预处理输入、warmup20、3×100、8threads float32的纯分类CPU速度；不要把旧34/50单次速度和18整图速度混比。可以在全GPU退出的安静条件下重测三架构整图速度，保留旧测量与重测理由。
4. 最终报告包含大对比表、混淆与每类指标、错误案例、检测瓶颈和A局限。16:50前停止训练，不用空跑填预算。
5. 视频3/P1音频待ASR；当前主机空闲RAM约3GB，不应同时加载大型CPUASR。优先主训练/比较；GPU释放后才排large-v3。先核实际语种，第二视频曾因中文标题误设zh，已保留初稿并用en重转纠正。

6. root已运行postprocess_A_results.py生成18/34混淆和曲线、comparison_results.csv/json（速度未齐有null，仅内部）。make_A_trained_gradcam.py已CPU实际执行，trained_gradcam/六行五列真训练模型矩阵及激活/梯度npz通过独审；不是旧ImageNet演示。error_examples.json与actual_crop_comparison.json在R根，图在report_figures，报告作者已纳入三图。18/34/50全齐后重跑汇总图表（不重新模型推理）。
7. documents依赖已loader，本批mark_artifact_operation_started已成功执行一次create expected4 docx（最初3份笔记+报告，后来第三视频扩大范围）；不要重复marker。root用ResNet双视频学习/build_notes_docx.py做Word，当前支持真实表格、cantSplit和科学图21.7cm高度，旧笔记不再重建覆盖；新报告另存P/A协议ResNet训练与对比实验报告_v1.docx，必须render全页独审后交付。

## 视频笔记与Word

目录P/ResNet双视频学习（P为MA6216_CA3_final_project）。已交聊天链接：ResNet视频一图文笔记.docx（12页16图）、ResNet视频二图文笔记.docx（11页18图）、ResNet整合图文笔记.docx（前两段整合14页16图）。原视频和音频均保存。

根build_notes_docx.py已处理中文、真实Word表、cantSplit、图片图注、去标题蓝线；终审_视频一、终审_视频二、终审_整合修表为最终规范渲染。notes_independent_review.json核视频1+前两段整合，视频2由root独核。

第三 https://b23.tv/Y6T5fii → BV1vb4y1k7BV，同济子豪兄4P，总3h17m12s。当前只下载P1 source.mp4（48876199bytes）及audio_16k.wav，1091.413sec，17代表帧、13裁图、OCR完整，均在视频3。P2-4仅metadata未下载。已向用户提供‘全部4P/仅P1’异步选择；用户回复前不下载其余P。agent resnet_video_one已完成媒体准备，无运行ASR。

## 最新完成记录：2026-11-02 16:09 +08:00

- 三种训练全部完成，最后ResNet50于16:03:42自然exit0；完整测试16:05:39结束、comparison_budget全阶段16:06:23 exit0。历史PID39260/4700/36452不再当故障重启。主控制session81336、对比49665均已结束。
- 18/34/50各24 COMPLETE TPE、18完整微调轮；三个逐样本独审PASS，模型/测试原始文件保留。50手图99.43835%、整图94.33962%，宏F1.99143994/.91366007，0误运动5/1773、stop误运动0/173但stop→0有32。
- 三模型统一纯CPU分类器计时300次：mean10.9737567/21.7412913/29.7657053ms；同100输入与权重身份，benchmark_A_classifiers.py已exit0。端到端CPU mean32.253655/45.88314/54.75644ms。comparison_results.json/csv与三模型学习曲线均更新；不得再用旧null摘要。
- 作者a_report_author正在最终补三模型大表/错误分析，root待生成P/A协议ResNet训练与对比实验报告_v1.docx并渲染独审。18可训参数不额外添加未知列，总参数三模型已实核。
- 全部训练/测试/CPU计时结束且GPU空闲才启动第三视频P1 large-v3 auto语种转写，exec session65882（18分P1，非全4P）。第三4P范围问题仍未回答；不下载其余P。独立准备稿视频3/p1_pre_asr_outline.md已实看13图，图06不能称完整架构、图09英文句首截断不全文引用。
- 个人research-assistant技能补长训前训练保存→真实推理工厂strict重载→真实尺寸前向的生命周期检查；官方UTF8验证PASS。

## 用户最新调整与交付

- 用户已放宽16:50时间限制，后续不再受原截止约束；已完成实验的旧日志/预算参数作为历史保留，不能倒改历史。三模型预定全部18轮和最终test均已正常完成，无待续截断轮次。
- 用户明确第三视频只做默认第1集，范围已锁定，不需要其他3P。
- 实验Word独审PASS，正文SHA097438753971018d03157f2229b95bfa8e90452fefe6a36bdd98a24052c673d6，最终Word SHAb6c42741203161bea665072055c0e744fde88e3b138a104ab12979fc6140da9f。已聊天交付P/A协议ResNet训练与对比实验报告_v1.docx；最终19页渲染在最终报告渲染_修字体。
- 新视频P1 ASR完整401段zh概率.999、1089.16秒语音覆盖至末，成功exit0。P1/三视频整合正文已完成独审待最终Word，根开始生成渲染；原两视频另存_v2修中文标题主题字体/表autofit，原v1保留不覆盖。

## 本轮最终完成

- A三模型全部预定训练/验证/测试/统一CPU测速完成并独审PASS；没有尚待续跑的部分轮次。用户放宽截止已登记，不改变已完成实验的历史参数/日志。B未执行。
- 实验Word：MA6216_CA3_final_project/A协议ResNet训练与对比实验报告_v1.docx；独审report_content_independent_review.json，19页9表6图。
- 三段独立视频笔记：ResNet双视频学习/ResNet视频一图文笔记_v2.docx（12页16图）、ResNet视频二图文笔记_v2.docx（11页18图）、ResNet视频三第一集图文笔记.docx（17页15图）。
- 整合：ResNet双视频学习/ResNet三视频整合图文笔记.docx（28页31图），覆盖两个原视频＋新增系列第1集；用户只选P1，无其余分集待办。原两稿正文/图片与v1完全一致，仅修字体与格式，旧文件保留。
- 新两Word 45页逐页独审PASS，notes_final_docx_review.json含固定MD/Word SHA及46处插图字节核对；旧两v2根逐页+正文/图字节审PASS记录notes_v2_format_review.json。所有render/ASR任务自然退出，无本轮训练/转写待监控进程。
- 最新有效Word渲染：实验R/最终报告渲染_修字体；笔记N/终审_视频一_v2、终审_视频二_v2、终审_视频三、终审_三视频整合。其它预览仅历史，不作为交付。

## 2026-11-02最新用户中止与B授权

用户要求立即停止当前交接/总包更新，到2026-11-03 02:30 Asia/Singapore之后自行完成B人员互斥实验，补项目方案/报告/README并重写交接和总ZIP；给组员不发图文学习笔记。已创建resnet-b ACTIVE每日02:30，详V/PROJECT_STATUS_B.md。旧含学习笔记的完整交接ZIP未交付且非最终，最新MD已新增组员后续工作但Word尚未更新，不再修旧稿。未启动B、原A不改。
