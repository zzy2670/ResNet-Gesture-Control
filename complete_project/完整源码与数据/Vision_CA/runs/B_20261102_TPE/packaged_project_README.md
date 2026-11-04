# ResNet手势识别项目

本项目输入一张完整RGB照片，先定位手，再用ResNet分类，最后输出0—5整数。1前进、2后退、3明确停止、4左转、5右转；0表示没有有效动作指令，不发起或续期运动，不能与3明确停止混同。A按照片划分和B按人员互斥划分下的ResNet18／34／50训练、冻结测试及CPU测速现均完成，下面分别保留各自数据和证据入口。程序负责图片识别，真实摄像头、无手场景及机器人控制联调仍待评价。

## 先看哪些文件

```text
Vision_CA/
├─ README.md                           本说明
├─ AGENTS.md                           项目代码规则
├─ prepare_datasets.py                 读取照片与标注、准备数据的 Python 程序
└─ data/
   ├─ raw_train_data/
   │  ├─ README.md                     HaGRID 30K 样本的数据介绍
   │  └─ hagrid-sample-30k-384p/        已接入真实数据目录
   │     ├─ hagrid_30k/                18类目录，共31,833张JPG照片
   │     ├─ ann_train_val/             18份JSON标注
   │     └─ annotations_df.parquet     标注表
   ├─ P1_random_image/
   │  └─ summary.json                  随机图片划分的统计记录
   ├─ P2_subject_independent/
   │  └─ summary.json                  按人员归档的旧统计记录
   └─ A_split_verified_20261102/
      └─ P1_random_image/              本次实际运行新生成的A分组
         ├─ summary.json               本次运行的统计
         ├─ train/                     25,468张训练照片
         ├─ val/                       3,185张验证照片
         └─ test/                      3,180张测试照片
```

以上列出当前项目能访问的文件。原工作目录的原始数据通过Windows目录联接接入，真实保存位置是同级 `HaGRID/dataset/hagrid-sample-30k-384p/`；完整复现包将入口保存为独立文件，迁移后不依赖该桌面联接。旧的 `P1_random_image`、`P2_subject_independent` 文件夹仍只有统计文件。本次另在 `A_split_verified_20261102/P1_random_image/` 实际生成A的三组照片、标注和逐图清单，保留旧记录不覆盖。新A使用独立照片副本，原始数据目录没有改变。

## 数据从哪里来

原始手势图像及标注来自 [HaGRID](https://github.com/hukenovs/hagrid)。课程使用的 [hagrid-sample-30k-384p 样本发布页](https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p) 由 cj-mills 提供，是从 HaGRID 缩小和抽取的样本，包含 31,833 张照片；不是重新采集的数据，也不是完整 HaGRID。当前 `raw_train_data/README.md` 保存了样本介绍。代码默认读取 `data/raw_train_data/hagrid-sample-30k-384p/`，本完整包在此路径保存原始数据文件，可在 VS Code 中逐层展开查看照片及标注。

## prepare_datasets.py 的原始数据准备逻辑

本节解释原脚本的实际行为。它的旧p2分支按人归档，后续正式B三组划分由独立的`prepare_B_datasets.py`完成；不能把旧p2文件夹与新B固定清单混为一谈。

1. `load_records` 读取照片，并按照片编号找到 JSON 标注。标注包含手框、手势标签和人员编号 `user_id`。
2. `make_image_split` 把照片分到训练、验证、测试三组；默认目标比例为 80%／10%／10%，随机种子为 42。结果供 P1 使用，同一个人可以出现在不同组。
3. `make_subject_split` 是按人员分成三组的函数。同一个人的全部照片应留在同一组，才能测试对新人员的识别能力。但当前 `main` 的 P2 分支没有调用这个函数。
4. 当前 P2 分支调用 `export_user_dataset`，按人员编号整理照片。这一步只是在“给每个人建档”，还没有指定谁属于训练、验证或测试。文件夹名含 `subject_independent` 本身不能证明三组划分已完成。
5. `export_dataset` 保存图片、标注和每张图片所属组的 `manifest.csv` 清单，并生成 `summary.json` 统计文件。

P1 的现有统计记录为训练 25,468 张、验证 3,185 张、测试 3,180 张。P2 的统计记录采用 `group_by_user_id` 模式，记录 31,833 张照片、7,973 人。统计文件中的 `/home/eddy/projects/vision_ca/...` 是生成记录时的源路径，不是当前 Windows 路径；这些记录不能证明其他设备上的目录现在是否完整。

### 对照注释阅读代码

先读脚本底部的 `main()`，它决定程序实际执行哪些函数；上面写有一个函数，不等于运行时一定会使用它。代码注释现已改为中文，下面给出整体阅读顺序。

```text
main()：读取运行选项和原始照片标注
  ├─ 选择 p1 或 both
  │    make_image_split()：把照片分到训练、验证、测试三组
  │    export_dataset()：把三组照片、标注、逐图清单和统计保存到磁盘
  └─ 选择 p2 或 both
       export_user_dataset()：按人保存照片、标注、逐图清单和统计
       此分支没有调用 make_subject_split()，因此没有把人员分到三组
```

`make_image_split()` 和 `make_subject_split()` 都是“决定照片属于哪组”的函数，返回的是程序内存中的分配结果。前者先在各个原始手势类别内随机打乱照片，再按约8∶1∶1分组；它不限制同一个人出现在不同组。后者先把人分组，再让每个人的全部照片跟随本人进入同一组；例如甲进训练组，甲的所有照片就都进训练组。后者的比例按人数计算，照片比例不一定正好也是8∶1∶1。

`export_dataset()` 与 `export_user_dataset()` 则负责“把结果保存成文件”。A会生成 `train`、`val`、`test` 三个文件夹，各有自己的 `manifest.csv` 逐图清单；原脚本的B分支会给每个人建文件夹，并在B根目录保存逐图清单。这份B清单描述照片属于谁、存在哪里，没有训练／验证／测试归属。两种导出都会生成名为 `summary.json` 的统计文件，但统计的是不同处理结果。

运行选项 `--ratios` 的默认值为0.8、0.1、0.1。程序会先检查比例是否合法，但当前只有A按该比例分组；选择 `--protocol p2` 不会自动调用按人划分三组的函数。脚本也不训练ResNet18、不把原始手势标签合并成0—5。

旧A、B文件夹只有统计文件，不能照着上述输出结构就认定旧分组数据已经存在。直接运行默认入口会因旧A文件夹非空而停止；`--extend-p1` 是在已有完整清单和照片基础上追加数据，不能只靠统计文件恢复原分配。

用户随后授权实际运行脚本。本次用新的输出根目录 `data/A_split_verified_20261102/`，选择 `--protocol p1`、随机种子42和比例0.8／0.1／0.1，已生成A的三组照片。每组都有 `images/` 照片、`annotations/` 标注及 `manifest.csv` 逐图名单。输入照片、比例、随机种子和代码相同时，分配是固定的；函数先在内存中分组，导出函数再保存这些分组。

新A训练、验证、测试组分别有25,468／3,185／3,180张照片，涉及6,999／1,876／1,931人。已逐张核对31,833张输出照片与源照片的哈希值，全部相同；两次重新计算的分组均与已保存名单完全一致。旧记录的三组照片数相同，但人数不同；因此这次是新生成的A分组，不声称恢复了原来那份缺失的名单。这里记录的是A初次准备；随后B已独立生成三组照片、标注和清单，详见下节。

## A阶段模型训练与测试

本节保留A阶段原始结果与当时的时延。A/B六模型当前速度比较采用后文同输入CPU补测，原时延在正式报告附录完整保留。

2026年11月2日已在本机RTX5060 Laptop 8GB上完成A训练和测试，直接复用核验过的逐图名单。ResNet18、34、50各完成24次TPE分类头搜索和18轮最后残差阶段微调；三模型共用的SSDLite手检测器完成16轮训练。最后一项训练于16:03:42正常结束，之后已完成验证集阈值冻结、完整测试、统一CPU测速及独立核查。完整照片经手检测、裁图、分类和多手规则输出0—5。这段记录A阶段；B的真实运行和结果见后文。

| 模型 | 人工框手图准确率 | 完整照片准确率 | CPU整图平均耗时 | 检测／分类阈值 |
|---|---:|---:|---:|---:|
| ResNet18 | 99.34% | 93.62% | 32.25 ms | 0.3／0.3 |
| ResNet34 | 99.18% | 93.14% | 45.88 ms | 0.3／0.5 |
| ResNet50 | 99.44% | 94.34% | 54.76 ms | 0.3／0.3 |

人工框分类使用3,917只手，完整照片使用3,180张；CPU整图测速使用8线程、batch=1，计入照片解码到数字汇总，模型加载不计入。三种模型共享检测权重，各自独立推理；公平比较分类器主要看相同人工框手图及统一纯分类测速。各模型的真实概率、框和错误样本均保留，不能拿人工框准确率代替完整照片准确率。

重要入口按以下顺序阅读：

- `train_A.py`：A主训练程序。`prepare`生成派生手图，`features`提取冻结特征，`search`使用TPE，`finetune`微调，`detector`训练定位器，`calibrate`只用验证集选阈值，`evaluate`完整测试，`latency`测CPU端到端时延。
- `run_A_budget.py`：监督主流程的阶段和截止时间，只管理本任务自己的进程，不关机、不重启电脑。
- `compare_resnets_A.py`：独立比较ResNet34与50，复用同一A名单、裁图和手检测器。
- `run_A_comparisons.py`：主流程结束后按剩余时间串行运行额外比较。
- `benchmark_A_classifiers.py`：使用相同预处理输入，单独比较三个分类网络的CPU速度，不把它混同整张照片的处理速度。
- `deployment_src/gesture_predict.py`：独立照片预测程序的代码模板，读取包内权重和配置，对整张图片检测、裁图、分类并输出0—5。
- `build_gesture_delivery.py`：把代码模板、三模型对应权重和阈值整理成分别独立的程序包。
- `audit_gesture_delivery.py`：检查交付文件身份、离线加载、实际图片预测和输出格式。
- `prepare_training_delivery.py`：把原训练结果与证据复制到统一训练记录目录，并逐文件校验SHA256。
- `runs/A_20261102_TPE/EXPERIMENT_PROTOCOL.md`：本轮固定规则、参数范围和测试口径。
- `runs/A_20261102_TPE/PROJECT_STATUS.md`：运行阶段与待办。实际结果以同目录JSON和原始日志为准，模型权重、逐样本预测及验证记录分别保存。

人工框手图分类的成绩只反映分类器；完整照片的成绩还包含漏检、框的位置、裁图、拒识和多手处理。测试集不用于训练或选择参数，不能在看到测试分数后继续调参却仍称独立最终测试。

## B人员互斥实验与新实验入口

B使用同一31,833张照片，按7,973名人员以随机种子42固定分组。三组分别为训练6,379人／25,274照片／30,799手、验证797人／3,457照片／4,326手、测试797人／3,102照片／3,759手。8∶1∶1按人数实施，照片比例为79.40%／10.86%／9.74%。三组人员、照片ID及照片内容哈希无交叉。A允许人员跨组，两套协议不要求相互排除照片。B从各自官方ImageNet初始化训练，没有继承A微调权重。

| 协议／模型 | 人工框准确率 | 完整照片准确率 | 整图宏F1 | 统一CPU分类均值／ms | 统一CPU整图均值／ms |
|---|---|---|---|---|---|
| A ResNet18 | 99.34% | 93.62% | 90.45% | 17.60 | 37.39 |
| A ResNet34 | 99.18% | 93.14% | 89.76% | 35.11 | 46.86 |
| A ResNet50 | 99.44% | 94.34% | 91.37% | 32.22 | 50.84 |
| B ResNet18 | 99.15% | 93.07% | 89.82% | 19.92 | 40.54 |
| B ResNet34 | 99.12% | 92.68% | 89.40% | 30.95 | 47.34 |
| B ResNet50 | 99.02% | 93.33% | 90.37% | 35.11 | 49.81 |

A人工框／整图测试分母为3,917只手／3,180张照片，B为3,759只手／3,102张照片。速度列来自一次预先固定协议的补充测量：六模型使用相同100只A验证手图和相同100张A验证完整照片，CPU8线程、float32，每模型每项20次预热、3轮300次调用，并非300张独立照片；纯分类不含预处理，整图含解码、检测、裁图、分类和汇总，每次一张照片，内部多只手按原接口成批分类。模型加载不计时，其他进程估计CPU占用为7.4%—43.0%，全部慢轮保留，不能据同架构的小差异判断加速。原有各协议测试输入的时延完整保留在正式报告附录，不用两次测量差值宣称模型改进，也没有据速度补测重算准确率或改模型。

B18真实0误发运动5／1,695，真实停止误发运动0／173，但停止输出0仍有41／173。B18首次评估中断后，在同冻结条件下恢复；两次3,759手预测标签和指标相同，1,039手概率变化、最大绝对差0.0010422468185425，未按测试结果改模型或阈值。B50按patience=4在11轮自然早停，使用第7轮权重；它的首次测试被系统重启中断，随后以同权重、阈值及清单恢复，具体概率差异见正式报告。B是同来源未见人员留出，不是新设备、真实机器人或背景变化条件的外部验收。各模型完整错误方向、停止漏识别与冻结条件见正式报告。

```text
Vision_CA/
├─ data/B_split_verified_20261102/
│  ├─ summary.json                 人员、照片、手框及互斥统计
│  ├─ train/                       25,274张照片及逐图manifest.csv
│  ├─ val/                         3,457张照片及逐图manifest.csv
│  └─ test/                        3,102张照片及逐图manifest.csv
├─ runs/B_20261102_TPE/             B18、B检测器与共享证据
│  ├─ compare_resnet34/            B34权重、搜索、日志与测试
│  ├─ compare_resnet50/            B50权重、搜索、日志与测试
│  ├─ trained_gradcam/             B18真实六类CAM及逐层张量
│  ├─ report_figures/              冻结预测派生的正式科学图
│  ├─ ab_comparison_results.json   六模型准确率、安全与耗时大表
│  └─ 实验报告正文.md              正式A/B报告正文
├─ portable_ab_project.py          完整包内只读核对数据与官方初始化
└─ portable_B_training.py          新目录中的显式单阶段实验入口
```

`prepare_B_datasets.py`负责实际人员互斥三组；`run_B_training.py`监督本次正式B各阶段，`run_B_pipeline.py`选择架构、B清单、初始化与输出；`postprocess_AB_results.py`只从既存预测和日志派生大表与图，不重推测试。`PROJECT_STATUS_B.md`保存当前项目状态，正式统计以冻结JSON和原始日志为准。原A与B的权重、清单、TPE数据库、逐样本预测和异常记录均保留。

### 使用已经训练的B模型识别照片

三个B独立包位于包根目录的`程序/B/resnet18`、`程序/B/resnet34`、`程序/B/resnet50`；A包位于`程序/A/resnet18`、`程序/A/resnet34`、`程序/A/resnet50`。每个包都有`predict.py`、`config.json`、两份权重、`requirements.txt`和自己的README。只需要识别图片时使用这些包，模型已包含，不需要重新训练。依赖安装可能联网，推理不下载权重。

从本README所在的`Vision_CA`文件夹，在PowerShell运行：

```powershell
Set-Location ../../程序/B/resnet18
python -m pip install -r requirements.txt
python predict.py examples/1.jpg
python predict.py "自己的完整照片.jpg" --json result.json
```

成功时标准输出只有一个整数。读图、权重或执行错误写标准错误并非零退出，不伪装成0；Python API的异常向调用方传播。多图调用可按包内README创建一次`GesturePredictor`后重复调用。握拳3优先，其余有效指令唯一则输出该指令，冲突或无有效指令则输出0；0类手不抵消另一只手的唯一有效动作。示例按验证照片ID固定选择，可能识别错误，合成空白图仅是接口示例。

本README位于解压包的`完整源码与数据/Vision_CA`。以下程序命令和文档链接均以这个文件夹为起点；包根目录README另外说明六个独立程序的选择和整体交接顺序。

### 在完整复现包中另建B实验

完整复现包的`Vision_CA`应同时包含固定A/B照片清单、原标注和`pretrained/checkpoints/`中的四个官方初始化；原工作目录的历史绝对路径和旧裁图索引不能直接用于另一台机器。先在完整包自己的`Vision_CA`打开终端，读取该包依赖说明并安装对应Python环境，再运行只读检查和帮助：

```powershell
python portable_ab_project.py check
python portable_B_training.py --list
python portable_B_training.py --help
```

确实需要一轮新实验时，首阶段使用尚不存在的新目录；下列只是单阶段入口示例，不是识别照片的步骤：

```powershell
python portable_B_training.py prepare --out runs/new_B --arch resnet18
python portable_B_training.py smoke --out runs/new_B --arch resnet18
python portable_B_training.py features --out runs/new_B --arch resnet18
python portable_B_training.py search --out runs/new_B --arch resnet18
python portable_B_training.py finetune --out runs/new_B --arch resnet18
python portable_B_training.py detector --out runs/new_B --arch resnet18
python portable_B_training.py calibrate --out runs/new_B --arch resnet18
python portable_B_training.py evaluate --out runs/new_B --arch resnet18
python portable_B_training.py latency --out runs/new_B --arch resnet18
python portable_B_training.py classifier-latency --out runs/new_B --arch resnet18
```

每条命令只执行一个阶段，默认分类搜索24次×18轮、末组微调上限18轮、检测16轮。`prepare`根据当前包照片重建新裁图与路径索引；34／50在同一个新`--out`下各自先执行`prepare`、`smoke`，再按自己的`features→search→finetune→calibrate→evaluate→latency→classifier-latency`接续，并复用新18裁图和新18检测器，不重复训练检测器。入口核对源码、manifest及官方初始化身份，拒绝写入原`A_20261102_TPE`或`B_20261102_TPE`，同一阶段不重复执行，并保留中断证据与串行锁。异常退出后需核已有阶段记录，不能删除锁后盲目重跑。原`run_A_budget.py`有历史截止时间，原B监督脚本与索引保留原运行身份；迁移复现使用上述新入口。

## 在 VS Code 中使用 Codex

在VS Code中打开`Vision_CA`文件夹，先阅读本README，再查看`prepare_datasets.py`的`main()`和训练入口。需要使用Codex时，使用`openai.chatgpt`扩展的“Codex”面板；“Chat with Copilot”属于另一个助手。工作区信任和登录状态由当前设备上的VS Code管理，README不把某次面板已打开的状态当成永久配置。数据目录有真实图片或目录联接时才可展开图片，仅有`summary.json`不代表图片已经包含在代码仓库中。


## 训练记录与独立预测包

完整的A与B训练证据保留在本目录的`runs/A_20261102_TPE`与`runs/B_20261102_TPE`，包括权重、TPE数据库、训练日志、验证冻结记录、完整测试逐样本预测及独审。旧JSON或日志中的原机绝对路径仅用于追溯，另建实验使用`portable_B_training.py`重新生成当前路径。

| 用途 | 包内入口 |
|---|---|
| A三个独立图片预测程序 | [ResNet18](../../程序/A/resnet18/README.md)、[ResNet34](../../程序/A/resnet34/README.md)、[ResNet50](../../程序/A/resnet50/README.md) |
| B三个独立图片预测程序 | [ResNet18](../../程序/B/resnet18/README.md)、[ResNet34](../../程序/B/resnet34/README.md)、[ResNet50](../../程序/B/resnet50/README.md) |
| 当前完整项目方案 | [项目方案v11](../../历史与正式文档/基于ResNet18的手势识别项目方案_v11.docx) |
| A与B正式实测报告 | [实验报告v2](../../历史与正式文档/A与B协议ResNet训练与对比实验报告_v2.docx) |
| A阶段历史实测报告 | [A报告v1](../../历史与正式文档/A协议ResNet训练与对比实验报告_v1.docx) |

六个程序各含`predict.py`、配置、检测与分类权重及README。识别照片按各包说明安装依赖并运行即可；无需训练，也不依赖原电脑的TorchHub缓存。模型加载后可复用同一Python实例处理多张照片。

`prepare_training_delivery.py`用于原工作区A阶段的训练证据精选副本，不是训练或推理入口。本完整包直接保留原运行证据和复现所需数据，不依赖原工作区的`ResNet手势识别交付/训练记录`。历史项目方案及理解问答用于追溯讨论过程，早期未训练、B未执行或接口说明以项目方案v11、正式报告v2及当前程序README为准。
