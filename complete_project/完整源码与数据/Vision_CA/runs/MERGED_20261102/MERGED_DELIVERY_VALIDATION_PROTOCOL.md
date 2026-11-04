# 合并六程序离线一致性核验

本轮只准备脚本，不加载模型、不推理、不构建ZIP。执行前必须等待六个新模型完整科学独审、十二模型公平CPU测速及其结果独审结束，并由root指定独占CPU窗口。

复用原 `Vision_CA/audit_B_gesture_delivery.py` 的 `check_package_files`、`check_hands` 和 `offline_cli_check`，新薄适配入口为同目录 `audit_merged_gesture_delivery.py`。不修改原工具、原A/B六程序、训练源、阈值或保存预测。默认调用或 `--help` 只打印计划。

固定样例由各协议val完整名单按image_id排序选择：六个单手六类、首张多手、四个单手raw手势 peace/peace_inverted/two_up/two_up_inverted，以及384×384 RGB纯白合成图。新class1四raw全部为1。每模型12次，六模型共72次源CPU与包CPU逐例比较；这是实现核验，不能把72例当新准确率benchmark。真实照片可识别错误，必须保留，不替换成功样例；白图按接口约定要求输出0。

同CPU比较按新merged源权重与新val阈值严格加载分类器、源协议检测器及包内模型。比较图尺寸、检测框数量与顺序、标签、接受决定、检测分数、分类置信度及六类概率、最终聚合。bbox容差1e-4，其余浮点容差1e-5；整数和接受决定必须一致。source用原 `train_A.whole_photo_predictions`、batch1，包用其实际 `GesturePredictor`。不与旧GPU数值要求逐位一致。

ZIP实际逐成员完整读取核CRC、SHA和字节数，目录manifest、当前文件集、当前文件SHA及ZIP一致。复用旧核验框架实际解压到中文含空格路径、外部中文cwd、空TORCH_HOME，通过子进程阻断socket/urllib/Torch Hub下载；核CLI stdout仅0—5加换行、JSON与源CPU一致、无预训练缓存生成。不存在图片必须非零退出、stdout为空、stderr非空。临时证据仅在新run的 `merged_deployment_validation`，不写旧B核验目录。

离线helper全部断言成功后，再完整核原程序目录/原ZIP及其身份不变，只清理本次mkdtemp目录下两个普通、非硬链接的解压权重副本 `中文 空格路径/ResNet*/weights/classifier.pt` 与 `detector.pt`。删除前独立核路径绝对包含关系、每层非软链/联接、manifest SHA/字节数，并立即再核目标SHA；不递归删除目录。保留manifest、config、samples、示例照片、predict、wrapper、详细结果、两次CLI原始stdout/stderr、清理计划及回执。回执说明提取副本曾完整验证后才删除，不能声称当前提取目录仍包含权重。原ZIP/完整程序不改。接口失败时全部解压证据保留，清理失败也保留剩余文件及已删除精确名单，不自动继续或重试。六份临时权重不再累计约0.503GiB，峰值约为单模型0.104GiB。

执行：

```text
<bundled-python> runs/MERGED_20261102/audit_merged_gesture_delivery.py --run-cpu
```

先于torch导入核实际 `aggregate_review.json` 的 `PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS` 与六逐模型审核及全部bound SHA。公平CPU结果必须为 `label_variants_cpu_benchmark.json`、status=complete且12模型；默认 `label_variants_cpu_independent_review.json` 必须status=PASS、无issues，且 `data_sha256` 等于实际CPU结果SHA；可用 `--cpu-review` 指定同schema的真实审核文件。此schema沿用原共同CPU结果审核，不创建假兼容PASS。

新六包输出 `merged_packages_independent_review.json`，status=PASS；`models`六项、每项12完整cases及离线证据；`archives`六项，每项至少有source、variant=合并、protocol、architecture、status=PASS、sha256、interface_verified=true、member_crc_sha256_verified=true、directory、directory_manifest_sha256；`bound_file_sha256`固定全部真实依赖。

开始实际模型核验后，独占 `merged_packages_independent_review.attempt.jsonl` 持久记录每例source/package完整结果和异常。已有结果或attempt拒绝无痕重跑。失败也保留部分结果，不删除、换样例或自动重跑。最终成功报告先序列化成准确字节，再核所有活身份，然后独占写同一字节并fsync。

六新程序通过后，单独执行：

```text
<bundled-python> runs/MERGED_20261102/audit_merged_gesture_delivery.py --combine-twelve
```

此步骤不重新模型推理。原A和B的审核schema不同：B沿用顶层identity_sha256；A没有该字段，使用原 `B_20261102_TPE/six_program_readiness.json` 锚定原A审核SHA、目录manifest SHA和三个ZIP SHA，再核原A各model的code/config SHA及offline_zip_run原ZIP身份，manifest中每个文件SHA成为当前复核依据。原A additional_root_validation仅保留为真实运行补充，不能当文件身份或全新pip安装证据。B亦需同一六程序readiness与原离线ZIP身份一致。重核十二程序目录与十二ZIP当前CRC/SHA后生成 `twelve_packages_independent_review.json`（status=PASS、12唯一archives），直接符合新总包builder门禁。旧程序文件不改。报告仍不是正式Word或最终大ZIP验收。

无推理小文件/schema检查可运行 `--check-legacy-schema`，不读权重或ZIP内容。当前空间方案见 `delivery_storage_budget.json`：root已决定最终大ZIP直接写已有E盘根目录 `E:/ResNet_合并与不合并完整交接_20261103.zip`，项目/文档/12程序仍留桌面。实际验证成功后再创建位置与SHA回执。
