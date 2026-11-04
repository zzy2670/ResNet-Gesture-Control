# 双标签十二程序最终ZIP独立验收入口

验收脚本为项目P根目录 `audit_final_label_bundle.py`，位于Vision_CA之外，防止本次最终审核器和结果自我纳入冻结循环。旧 `audit_final_ab_bundle.py` 不改；仅复用其ZIP安全路径函数，不调用其旧六程序main或consumer推理。

当前默认只打印PREPARED_NOT_EXECUTED。AST、help、默认入口及小型内存链接结构检查不代表实际总ZIP验收，当前没有读取最终大ZIP、没有执行大散列或任何模型推理。

最终命令：

```text
<bundled-python> audit_final_label_bundle.py --run
```

默认读取 `E:/ResNet_合并与不合并完整交接_20261103.zip`，其旁的 `.verification.json` builder实际回执，以及桌面P下同名总包根的SHA256SUMS、冻结成员映射和来源说明。可用--archive/--bundle/--receipt明确指定实际路径；根名仍须与正式根名一致。未实际生成的ZIP、receipt或审核不可用任意PASS占位。

前置证据包括源码工具预审（当前SHA必须一致）、真实合并六模型aggregate/逐模型审核、原A/B历史审核、新正式三Word独审、十二程序离线接口/ZIP审核、十二模型公平CPU结果独审、合并A/B新CAM科学独审。默认CPU独审文件 `label_variants_cpu_independent_review.json` 要求status=PASS且data_sha256对应实际CPU结果；新CAM独审默认 `merged_gradcam_independent_review.json`，要求status=PASS、非空bound_file_sha256，明确绑定A/B新CAM各六NPZ+PNG+metadata的实际源字节。可用--cpu-review和--cam-review指定相同真实schema的文件，不伪造兼容审核。

本轮独立流式读大ZIP的每个实际成员到底，重算SHA、字节数和CRC，核成员完整集合、大小写重名、路径逃逸、文件类型和冻结清单/映射。本轮自己核外层原始archive SHA与builder receipt，成功报告编码后再重核archive SHA和外部证据。builder receipt只作与同一不可变archive SHA绑定的交叉证据，不代替独立成员检查，也不重复运行作者verify增加一整轮大ZIP读取。

十二程序目录逐文件与各自manifest一致；逐一核配置、协议/架构/标签版本、新旧科学权重/val阈值、十二索引和已真实离线测试的ZIP SHA。十二小ZIP在内存中逐个打开，独立完整核内部成员SHA/CRC/集合，与大包内对应目录完全一致；不落盘重复解压大权重。既有真实CPU离线测试的ZIP身份若与总包实际相同，本轮不再重复十二模型推理。不能把既有离线PASS当作大ZIP内容已核。

同时核真实raw/A/B各31833照片、必要v12/v3/根交接、旧v11/v2和课程原文；个人图文学习笔记、音视频/字幕转写、可重建裁图/features缓存不收，原A/B及新A/B科研CAM必须保留，旧中断/恢复/原CPU慢轮证据保留。

所有Markdown和Word均扫描实际链接。Markdown覆盖行内/引用式链接与图片和HTML href/src，排除代码文本；Word内部媒体/关系使用OPC包成员解析：内部前导 `/` 指DOCX内部根，dot-segment规范化后必须仍在包内且成员实际存在，跨包根的 `..`、磁盘scheme、网络authority和反斜线内部目标拒绝。`TargetMode=External`始终按外部本地/网络规则分类，不能借内部根规则放行C盘原机路径；其相对本地目标再按大ZIP真实位置解析。正式入口root README、交接MD、打包源码README、十二小包README，以及新正式三Word的本地目标必须实际可用，绝对原机链接或断链会阻断验收。原/历史MD与Word的原机绝对链接、旧相对断链完整逐项列为来源限制，不改写原科学字节，也不宣称所有历史资料均可移植。外部HTTP/app链接不联网，同文档锚点名称不做渲染器级验证。

唯一最终结果 `Vision_CA/runs/MERGED_20261102/final_bundle_independent_review.json` 独占创建，status=PASS表示本定义范围完成；包含archive路径/SHA/长度、全部成员验证、十二目录/小ZIP身份、科学/正式文档证据、链接完整清单和历史限制、旧异常证据与实际消费边界。不覆盖既有结果，不训练、不推理、不重算准确率、不重测速度、不改来源。若未完成或失败，不生成PASS；原ZIP及失败证据保留。

冻结后外部验收报告在R产生，不能再把它加入刚完成的同一ZIP；其SHA/结果和E盘位置可在最终桌面项目回执中记录。真正执行仍由root在所有真实产物固定后安排。
