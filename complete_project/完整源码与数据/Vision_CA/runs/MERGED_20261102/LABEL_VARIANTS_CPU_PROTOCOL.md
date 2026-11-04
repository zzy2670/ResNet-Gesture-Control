# 双标签版本十二模型共同输入 CPU 速度协议

本协议在十二模型补测前固定。旧未合并六模型训练、准确率、阈值和共同 CPU 基准全部保留。本轮只比较新旧标签版本 × A/B 协议 × ResNet18/34/50 的实际速度，不重算准确率、不调权重或阈值、不选择最快轮次。合并版本第1类包含 peace、peace_inverted、two_up、two_up_inverted；未合并版本仍使用原映射，其余0/2/3/4/5类含义不变。两版本都保持六输出。

唯一共同输入来源是旧 `runs/B_20261102_TPE/matched_cpu_benchmark.json`，冻结 SHA256 为 `23f8dd1e2b08ac7e9e2f4f1618534212a209f867410a54873663555a6e3fba68`。直接使用其中 `photos` 的100张 A-val 完整照片和 `classifier_inputs` 的100张 A-val GT 手图，保持原顺序、keys、源照片/裁图 SHA256 及预处理张量 SHA256。禁止重新选择前100个、按结果换样本、改像素或只复用检测缓存。读取原 A-val 元数据只用于通过旧ID定位照片及原始标签。两版本分别从 raw_labels 重算元数据 labels/truth，避免把旧truth冒称新truth；此truth不计准确率，也不参与速度筛选。

新模型来自 `runs/MERGED_20261102/{A,B}/{resnet18,resnet34,resnet50}`，未合并模型来自原 A/B 目录。所有分类器按明确架构离线构建并严格加载实际 checkpoint；检测器使用原部署源码的 full-tail SSD320 离线工厂严格加载。每个新模型的 detector SHA 必须与同协议、同架构旧模型一致，旧模型双权重必须与旧共同基准一致。每个模型使用自己仅由验证集冻结的检测和分类阈值，不能统一真实检测框数或接受数量。原权重、校准、参数、标签身份和调用源码均记录SHA。

固定 CPU、float32、8个 PyTorch 算子线程。一次调用为一张完整照片或一个 `[1,3,224,224]` 分类输入。纯分类在计时前使用原 RGB/ToTensor/ImageNet Normalize 得到张量，并与旧tensor SHA逐项核验；只计分类器前向。整图直接使用原 `train_A.whole_photo_predictions(..., rows=[row], batch_size=1)`，包含 PIL 读图解码、检测、真实裁图、分类概率和指令规则处理。照片内多手仍按原接口一起分类，内部手批量可能大于1；不能误称所有内部操作均batch1。加载、哈希和输入预处理均在计时外。

全部十二模型提前加载常驻内存，预期进程约3GiB，适配本机32GiB内存；加载前要求可用内存至少6GiB，并记录实际加载前后与每模型加载后RSS/可用内存。此预算是资源预检，不是已实测用量。每个scope先为每模型预热20次，然后3轮，每轮每模型固定遍历全部100个输入。先纯分类scope，后整图scope。

基础顺序严格固定为：未合并A18、合并A18、未合并B18、合并B18、未合并A34、合并A34、未合并B34、合并B34、未合并A50、合并A50、未合并B50、合并B50。第2轮完全逆序；第3轮基础顺序循环左移6位。三轮只能部分平衡持续漂移，不能消除系统背景负载；每模型每scope全部300次毫秒时间都保留，共72个计时块、7200次正式调用。所有慢轮与背景负载证据保留，不剔除异常慢值，不重复测速择优。

每个计时块记录开始/结束时间、100次原始ms、均值、系统及本进程CPU计数、内存、其他进程忙碌占比估计；每次整图记录固定阈值后的检测框数及分类接受手数，说明实际工作量。汇总包含 mean、p50、p95、样本标准差(ddof=1)及三轮均值。背景负载和模型阈值引起的真实工作量差异须用于解释结果；同架构标签版本间小差异或非单调深度排序不能被当作训练语义/深度导致的因果加速。

执行必须等待新六模型训练、完整评价和科学独审全部完成，再由协调者明确提供独占 CPU 窗口。程序在任何模型加载前强制核验真实 `all_training_evaluation_complete.json` 的 `status=PASS`、`aggregate_review.json` 的 `status=PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS`及空issues、六个真实 `model_review.json` 的 `status=PASS_MERGED_MODEL_AUDIT_WITH_LIMITATIONS`及空issues与协议/架构身份。汇总必须唯一覆盖六模型并绑定实际逐模型独审；逐模型独审必须绑定双权重、校准、val原始输出及三个test原始输出，选定阈值/双权重与实际校准及汇总必须一致；独审全部绑定文件的SHA在加载前复核。权威完成/独审及其全部绑定身份写入结果，并在materialize之后再次precommit复核。缺失或变化直接保留失败日志并停止，不创建虚假的兼容PASS文件。当前准备阶段只运行 AST、`--help`及默认打印协议，不加载模型、不执行CPU推理，也不把AST当依赖预检。正式 `--run` 完成上述门禁后实际导入 torch、torchvision、psutil并记录版本，要求psutil为7.0.0；没有完成实际导入不能声称依赖通过。协调者应使用已验证解释器并提供 `PYTHONPATH=E:/python/Lib/site-packages`。

新脚本为 `benchmark_label_variants_cpu.py`，默认只显示本协议，明确传 `--run` 才执行一次。新成功结果为本目录 `label_variants_cpu_benchmark.json`，独占尝试日志为 `label_variants_cpu_benchmark.attempt.jsonl`；已有结果或尝试日志则拒绝覆盖或自动重跑。每个计时块立即持久化，异常保留错误与部分原始证据。最终结果先materialize为字节，再在独占创建前重核十二模型双权重/校准、参数/标签身份、代码/协议、旧参考结果、索引及选定照片/手图/source SHA，随后提交同一份字节并fsync；所有提交核验在计时外。原基准脚本、原六模型 CPU 结果及训练 pinned source 均不修改。
