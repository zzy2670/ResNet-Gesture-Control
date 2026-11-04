# 合并与不合并实验的最终验收清单

本清单按用户新增的“两种标签方案都训练并重新打包”执行。训练完成不等于整项任务完成；所有正式文档与压缩包均须依真实结果验收。

| 要求 | 权威证据位置 | 当前验收状态 |
|---|---|---|
| 保留不合并 A/B × ResNet18/34/50 的训练、完整测试及失败记录 | `../B_20261102_TPE/ab_final_independent_review.json`、原 A/B 运行目录 | 既有结果保留；最终封包时再核冻结字节 |
| 原 A 图片划分和 B 人员互斥划分不变，标签合并只改变两种 two_up 的目标 | `preflight_data_independent_review.json`、各模型 `merged_data_identity.json` | 数据预检已完成 |
| 新六模型使用各自协议/架构原定参数；不新增 TPE、不使用旧微调分类器作初始化 | `preflight_independent_review.json`、六模型 `model_review.json` | 源码预检通过；逐模型实际审计进行中 |
| 合并 A–ResNet18 完整训练、阈值冻结、完整测试及逐样本审核 | `A/resnet18/model_review.json` | 已通过，有明确适用范围和限制 |
| 合并 A–ResNet34 完整训练、阈值冻结、完整测试及逐样本审核 | `A/resnet34/model_review.json` | 已通过，有明确适用范围和限制 |
| 合并 A–ResNet50 完整训练与审计 | `A/resnet50/model_review.json` | 已通过，243035项检查、0问题；保留范围限制 |
| 合并 B–ResNet18/34/50 完整训练与审计 | `B/resnet18/model_review.json`、`B/resnet34/model_review.json`、`B/resnet50/model_review.json` | 三模型均已完整训练、冻结测试并独审通过 |
| 六模型全部自然完成、汇总独审及全部绑定文件身份一致 | `all_training_evaluation_complete.json`、`aggregate_review.json` | 已通过，保留科学范围限制 |
| 新旧任务语义、各自分母、四种原始两指手势及共同负类成对比较 | 六模型 `model_review.json` 的 `paired_original_vs_merged` | 已完成模型已核，其余待完整输出 |
| 十二模型使用相同输入、CPU线程、测量范围与固定次数公平测速 | `label_variants_cpu_benchmark.json` 及相应独审 | 真实测速及独审通过：72块、7200原计时、24汇总 |
| 新合并 A/B 代表 ResNet18 的分层 CAM、原值、输入身份与呈现独审；保留旧图语义 | `merged_gradcam_independent_review.json`、各 `trained_gradcam/` | 已通过：12 NPZ、48层重算及2张完整PNG像素一致，16项新产物绑定 |
| 新合并六程序具有真实新权重、阈值、原始标签映射和固定样例 | `build_merged_gesture_delivery.py` 的真实构建产物 | 已构建并经实际审核 |
| 新六程序完成实际离线解压、中文空格路径、无缓存/禁网、接口与同源 CPU 一致性检查 | 合并交付独审、`audit_merged_gesture_delivery.py` 实际输出 | 72例一致性及离线核验通过，保留5个自然预测错例 |
| 十二程序目录当前内容、清单与十二个独立 ZIP 一致 | 十二程序汇总审核，逐目录与 ZIP 成员 SHA/CRC | 已通过；最终大ZIP中的同字节身份仍须验收 |
| 项目方案 v12、实验报告 v3、README 真实更新；保留必要历史 Word 与阅读顺序 | 新版 Word、README、文档独审及渲染产物 | 已通过三Word共104页独审，数字/旧图表/链接均核 |
| 交接说明讲清共同 ResNet 分工、组员后续角色与产物，不虚构既定分配或课程验收 | 新总包根目录 `交接说明.docx` 和 Markdown | 已通过独审：八章、Word9页；课程整体不冒称验收 |
| 讲清 SSDLite 找手、ResNet 单手分类、额外规则合并输出的职责，保留每手细节且 CLI 只输出一个数字 | 新文档与实际推理接口 | 源码、实际接口及正式文档独审均通过 |
| 双手错误抽查使用原图/GT/冻结预测，区分漏检、框偏移与分类错误 | `detection_spotcheck_20261103/spotcheck.json` 及独审、对照图 | 四张指定样本已核；不代表总体发生率 |
| 给组员的最终包排除个人学习笔记、视频、音频与转写，保留科研 CAM、原始实验异常和必要课程材料 | 最终冻结成员清单、排除项审核 | 未封包，不能提前声称已排除 |
| 新总 ZIP 完整包含两种标签方案 × A/B × 三架构及必要数据/代码/文档 | `ResNet_合并与不合并完整交接_20261103` 对应最终 ZIP 与冻结清单 | 未生成；旧中止草稿不得交付 |
| 固定全部文件后封包，逐成员 SHA/CRC、链接和实际解压入口验证通过 | 新总包封包回执与独立最终审核 | 未完成 |

旧 `ResNet_AB项目完整交接_20261102` 的冻结与中止 ZIP 属于扩大任务前的历史，不作为本表最后两项的证明。当前唯一 GPU 控制器保持运行；不为更新本清单重启任务或重跑测试。
