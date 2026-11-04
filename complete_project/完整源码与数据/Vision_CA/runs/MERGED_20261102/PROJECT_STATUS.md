## 合并版真实GPU已启动 2026-11-03T00:00:29.742922+08:00

root已核preflight_independent_review.json=PASS_PREFLIGHT，于2026-11-02 23:56:09启动唯一controllerPID27144，ROOT_GPU_LAUNCH.json保存实际创建时间。最新阶段A/resnet18/finetune，childPID34868，日志C:\Users\Administrator\Desktop\6216_VISION\MA6216_CA3_final_project\Vision_CA\runs\MERGED_20261102\A\resnet18\stage_logs\finetune_attempt1.log。A18单固定参数head已完整18轮exit0，不含任何TPE搜索；后续真实微调最大18/patience4，root已见A18第2轮全量31095手完成且GPU94%/1910MiB。CPU烟测6组与数据/总preflight全PASS，不能把本状态当六模型结果已完。

新并行准备：merged_independent_preflight写只读audit_merged_results.py/日后模型审核，不碰GPU；ab_package_builder准备新build_merged_gesture_delivery.py/build_label_comparison_bundle.py（原6程序和oldbuilder不改）；ab_plan_handover_finalization高模型只准备v12/双版本交接facts与新生成器，不覆盖旧Word。正式12程序/比较报告与新ZIP必须等六merged训练/完整新测试/独审/共同速度+raw子群对比后。

本目录PROJECT_STATUS由root继续追加，旧PROJECT_STATUS_B顶指向此处。保持active goal与resnet-b ACTIVE，不重开健康controller，不为占满显存重复模型。若退出先核实际PID创建时间/exit/log和本机系统重启事件，原日志保留，独审后受控恢复，不能盲目重跑固定test或TPE。

# 合并与不合并对比：当前有效状态

用户2026-11-02在6212聊天明确转交本聊天：原不合并已完成，新增真实合并训练后再统一打包。旧封包进程32112已停止，未交付最终ZIP；已对.generated partial留size/SHA回执后清理临时输出。旧模型/划分/全部科学输出/方案v11/报告v2/交接及冻结清单保留。原最终封存提示不代表新目标完成。

新合并标签四raw peace/peace_inverted/two_up/two_up_inverted→1，其余映射保持；只有两种two_up相对原标签0→1。原A照片split与B人员split、图片/手框/像素不重分。新run为本目录A或B/resnet18、34、50，GPU唯一root串行。各自原手检测器可复用；不重做TPE，沿用各协议架构的已选params，官方ImageNet特征新标签重新训练FC18轮，再layer4+fc最多18/patience4；不得直接复用旧已微调分类器或仅后处理输出0→1。

数据writer merged_data_preparation已完成六run索引、重新聚合photo truth、新labels cache与完整identity；preflight_data_independent_review.json PASS。A/B手图test class0新支持2149/2016，完整照片command0新支持1412/1359，分母严禁混用。六CPU真实classifier strictload+forward/backward、各自detector strictload+forward均已完成all_cpu_smokes_complete.json。merged_independent_preflight正在最后核源码/marker/输入；最终preflight PASS后root才启动GPU。当前尚未开始GPU分类训练。

sole写入：数据prepare及新索引/cache由merged_data_preparation；pipeline/controller/MERGED_PROTOCOL由merged_training_pipeline；最终preflight JSON由merged_independent_preflight；本PROJECT_STATUS与GPU启动由root。旧A/B及旧原始报告不写。后续合并六模型val阈值冻结完整test/各类原raw子群与共同负类安全指标、统一公平速度/CAM/错误分析、分清目标语义变化后，另存方案v12/报告v3/新版交接和12程序总包，全部独审再归档验收。

原goal保持ACTIVE并包含以上最新扩大要求；resnet-b心跳已更新ACTIVE，每30分钟按本状态推进，不重复启动健康进程、不重启关机本机、不操作桌面或无关项目。


## root当前实核与并行准备 2026-11-03T00:18:22.376183

唯一controller PID27144与A34微调child PID33348均已核实际进程创建时间和CPU增长，GPU98%、2303MiB；A34第12轮完整31095训练手于00:17:29完成。A18已完整训练、冻结测试、自然exit0，model_review.json为PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS、242867检查、0问题。A18新整图准确率91.9497%，人工框99.0554%，不得以合并必然更好描述，也不按该test结果调参。旧文中尚未GPU的段落为历史，真实启动时间23:56:09。

已分配ab_report_finalization准备全12模型对照v3报告新生成器（旧v2只读）；merged_independent_preflight复审新打包工具门禁修复，随后A34测试完成才逐模型CPU审核。v12/新交接生成器与12模型CPU公平测速工具已准备，不生成缺失数据的Word、不提前封包。root继续唯一GPU调度；所有合并6模型、完整新测试、科学独审、共同CPU速度、12程序真实离线验证、正式文档与总ZIP核验齐备前目标保持ACTIVE。


## root推进 2026-11-03T00:25:04.693661
A34完整evaluate于00:23:53自然exit0；唯一controller27144随后启动A50fine child23728于00:24:04。已分派merged_independent_preflight实际单线程CPU核A34完整输出，尚不预宣通过。新两打包工具源码独审PASS_DELIVERY_TOOLS_SOURCE_PREFLIGHT_WITH_LIMITATIONS，五问题关闭，文件SHA62233f2d69d4922c12552a1939f202f3a4b3362f1d8f82850513f18baa82c0b3；这不是十二程序/ZIP产品通过。root已逐段二审科学审核源码，audit_logic_root_review.json明确范围限制。
用户要求的四张B18双手照片抽查已完成并由独审核原图/原标注/冻结预测/IoU/对照图：两手齐1例，全漏1例，漏fist只找到垂手1例，peace已定位却分错且另手框偏1例。材料detection_spotcheck_20261103；不遍历图片、不重推理、不调阈值、不由四例估总体率。图p为检测分数；数字0正确也不能证明手都检测到；分类错误不得单凭图归因于框偏移。v3报告作者已收证据及说明，v12/交接作者补三组件职责。


## root验收与存储决策 2026-11-03T00:35:46.765222
A34独审已真实通过243275项、0问题，model_review SHA792468c77a28b4896eb05d1c3f6fc0eff2a5ce99c80c4ea0f85bf968b3f4da2a。A50fine PID23728仍活且第7轮全量完成，controller27144保持。
新合并离线核验薄适配已准备，但root实读旧A交付独审JSON发现其无顶层identity_sha256，新combine_twelve把A/B假定同schema会KeyError；已交原独审writer修复，不能提前当产品通过。旧A六程序身份应结合各model offline_zip_run与B/six_program_readiness当前档核，不以空dict略过。
磁盘只读预算：C当前余约7.5GiB，后续新增包/临时验证/总ZIP保守约8GB且还有B模型增长，不能挤占训练空间；E约69GiB可用。root决定最后大总ZIP通过原生--archive输出E:/ResNet_合并与不合并完整交接_20261103.zip，桌面P仍保留正式文档/项目/十二程序和总包清单，验包后加位置与SHA入口回执。不删原A/B/任何旧模型或数据，不以cleanup未知用户文件来腾空间。新核验专属解压副本仅成功后可按确定生成/同源SHA/路径范围清理大权重，失败留证据。当前尚未生成此总ZIP。


## root实核：合并A三模型全部已审 2026-11-03T00:46:55.828552
[{"arch": "resnet18", "sha256": "04b313b5026a7c2bb4e61e1fd8d2c06d2453495d453389a2c8c75aec3ea6b975", "checks": 242867, "oracle_accuracy": 0.9905539954046464, "whole_accuracy": 0.919496855345912}, {"arch": "resnet34", "sha256": "792468c77a28b4896eb05d1c3f6fc0eff2a5ce99c80c4ea0f85bf968b3f4da2a", "checks": 243275, "oracle_accuracy": 0.9890222108756701, "whole_accuracy": 0.9216981132075471}, {"arch": "resnet50", "sha256": "01051709d057932cc3c76db02dd69c2e7fb5c6d4b7f1cdae0c85fb2354b1e268", "checks": 243035, "oracle_accuracy": 0.993617564462599, "whole_accuracy": 0.9289308176100629}]
A50于00:45:11完成evaluate自然exit0，独审243035检查0问题；训练head18、fine16/pat4/best12，不能把早停误称只跑部分。随后唯一controller27144于00:45:20启动B18fine PID10916（root实核创建时间和CPU增长），B三模型仍待完成。新A18/A34/A50整图准确率分别91.9497/92.1698/92.8931%，各自新语义下结果，不能按新测试表现改参数。
新CAM源码及协议已独立预审通过（merged_gradcam_preflight_review.json），仅准备未模型加载/推理；正式CPU入口应在六模型/科学总独审/12共同速度及独审之后由root依实际CPU窗口发限时permit，不需要用户批准。合并离线adapter已修真实A/B schema并经root源码复审与实际小文件schema测试通过（merged_delivery_adapter_root_source_review.json）；未来72例实际接口及12ZIP仍未执行。总验收清单见COMPLETION_CHECKLIST.md，不能将已准备入口当最终产品完成。


## root实核：合并B18已完成并独审 2026-11-03T01:00:20.294075
B18 evaluate于00:58:12自然exit0，科学独审238413项0问题、SHA563e5d99ddfbc360b7da2d4ec207335cfd808fae74657cc1752a799107ccd284。head18/best13，fine17/pat4/best13；GT3759准确99.0689013%，整图3102准确91.8439716%。整图0→运动16/1359、停止→运动0/173、停止→0=40/173。共同纯原始负类手图误运动旧4→新11/2016，整图旧5→新16/1359；新旧任务定义不同且共同负类风险增加必须保留，不能宣称合并必然更优或以test改参数。
唯一controller27144继续，B34fine PID38120于00:58:22启动，root已核进程创建时间和CPU增长；B50随后，仍有两个合并模型未完成。12模型共同CPU数值独审器audit_label_variants_cpu.py已准备并由root非作者核实际旧schema与纯数值测试通过（label_cpu_audit_root_preflight.json）；没有执行真正共同测速或写真实数值审核PASS。所有正式文档/12程序/大ZIP仍等后续完整证据。


## root实核：合并5/6已审 2026-11-03T01:15:43.189385
B34完整训练head18/fine18/best17、冻结test自然exit0，逐模型独审238659检查0问题，SHA57a98b4ee197f5962577d395836e5a498e768390b324611d61030a1991301464。GT3759准确99.0689013%、宏F1 .991675；整图3102准确91.8117344%、宏F1 .893644。整图0→运动14/1359、停止→运动2/173、停止→0=42/173；共同负类手图旧5→新12/2016、整图旧9→新14/1359。整图和手图的指标方向/分母保持分别报告。唯一controller27144随后于01:13:53启动最后B50fine child30956；root继续核实际进程，健康任务不重启。
新源码包README已完成并经root非作者内容/逻辑路径核查：packaged_project_README.md SHA df989331f01b2f0d2aca86017334ed3b5ebcaf8b3df6a4661b20fa85bbd11202，packaged_README_root_review.json记录18个预定ZIP相对目标，实际成员仍须最终封包验收。未改变旧README/Word；当前环境与历史环境分开，不假称完整源码controller已一键移机，便携入口为各独立推理包。正式v12/v3及根交接尚未生成，仍等6/6科学总独审、12共同速度/独审、真实新CAM/独审、12程序实际离线核验等证据。


## root实核：六合并科学完成与唯一CPU测速启动 2026-11-03T01:35:30.176472+08:00
最后B50完整测试自然exit0；6/6逐模型和aggregate均已真实独审，aggregate SHA e433fbfd18ac3cd28a6c3e22f7f7e7f5b728347976267a3c5e8789b5571da5af。controller27144已自然结束，不能重启它。root已启动唯一benchmark_label_variants_cpu.py --run，exec session39626、实际PID39600及创建时间见ROOT_CPU_LAUNCH.json。CPU窗口不并行模型/CAM/离线程序审核/大hash。尚无真实速度最终结果或独审；健康测速只读不重启。


## root实核：速度/十二程序已审，正式文档并行 2026-11-03T01:50:39.426968+08:00
12CPU唯一测速session39626已自然exit0，真实JSON SHA f78457295f9be97f5a4bb2452c312232f5c773db4cfa95c8c25cf4cf9c820813，数值独审PASS、0issues，审核SHA e6d6ed0bfbc39b2939a0ff956e72ef2c71815657f128493cf9a47f3885cf9f75，72blocks/7200原ms/24summary全部重算。无需重开测速。
新CAM A/B真实生成均exit0（会话6591/60822），两个新PNG/12NPZ/metadata保持原值；merged_independent_preflight正在实际单线程数学和原像素看图审查，未预宣科学PASS。B图缩放预览像缺字的现象经original视图与像素墨迹核已确认为预览显示问题，不改任何图或重推理。
六新程序build session97454 exit0，72例实际CPU source/package及离线核session71438 exit0 PASS；5个自然识别错例保留，不能称72全识别正确或新准确率。combine12 session56261 exit0，twelve_packages_independent_review.json PASS SHA b2e94faa37d15a0ae40f2db9fa84cd831c098d52acb9e03b9119f30adb3add08，原六包未改。新临时解压权重仅成功后验证同源字节清理副本，完整原程序/ZIP和诊断记录保留。
高模型ab_report_finalization独占真实解释稿/report_payload与v3报告；高模型ab_plan_handover_finalization独占最终facts/v12/新根交接，双方已协作，等CAM真实审及报告payload齐后才不可变FINAL facts和正式Word。root状态为操作日志持续更新，不纳入facts静态源。P/audit_final_label_bundle.py已准备并经root非作者源码/OPC小例预审，最终E大ZIP尚未生成/验收；任务不完成。


## root实核：科学/速度/程序/CAM全齐，待正式文档与总ZIP 2026-11-03T01:55:08.663107+08:00
新CAM真实独审PASS SHA 46ab163eae61bd98a2b1cbbf866d5595ed2ae9dbaf7ce977343b3af84bd1a975；root实读status/0issues并逐一核A/B共16产物bound实际SHA。旧图未改，48层数值和2PNG原像素全重建一致。至此六新模型科学、共同12CPU、十二推理目录/独立ZIP及实际72例新包离线一致性、两新CAM全部真实完成。后续不得重跑这些健康已审任务。
两高模型作者正在实际写解释稿与完整12项结论并汇编不可变facts，正式v12/v3/新根交接Word随后生成和逐页独审；当前尚未宣称正式文档通过或E大ZIP已生成。root已给作者真实review路径/SHA，操作日志不作facts静态源。新总ZIP独立auditor在P/audit_final_label_bundle.py，其root源码预审PASS，实际大ZIP验收仍待执行。


## root封存前最后状态 2026-11-03T02:13:23.961905+08:00
全部6新模型科学/12CPU/12程序/新CAM/三正式Word与README均已实际独审。三Word共104页，正式独审SHA192279a597db85500ae7b172a55d73d4304faa2da28b1b9e709fca1de1d5844d；v12 f9f136d40aac91325732a31bd3586409be9a5c9de989805fc30a706c0443d5bb，v3 f8071465cb0742ee86dd4e2f9c8c05481d491f00ccc53b319710001b79ae8b28，交接87d35dac652b5d0bafc460818c3dadc36e637479e2ac23d75633d701efabbb23。所有作者已固定版本，PREPARATION草稿和排版缓存留包外或明确忽略目录。
现在冻结白名单实际来源并流式生成E:/ResNet_合并与不合并完整交接_20261103.zip；仍须完整成员CRC/SHA、12小包身份、实际文档链接和独立总包验收通过才能整体验收。该状态随本次封存快照保存；封包后的独立验收报告及E侧verification回执为额外外置证明，不能先宣称总ZIP已通过。请勿重开任何训练/测速/程序重复审核。
