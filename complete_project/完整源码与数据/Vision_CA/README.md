> [!IMPORTANT]
> **当前仓库入口**：[仓库首页](../../../README.md)、[交接说明 v3](../../../current_bilingual/documents/交接说明_v3_中英文配图.docx)、[综合实验报告 v4](../../../current_bilingual/documents/A与B协议ResNet训练与对比实验报告_v4_中英文配图.docx)、[项目方案 v13](../../../current_bilingual/documents/基于ResNet18的手势识别项目方案_v13_中英文配图.docx)。下文保留原包源码说明，报告 v3、方案 v12 及原交接 Word 为历史入口。
>
> 十二个可运行程序已在[../../程序/](../../程序/)展开；下文相对程序路径仍适用。下文提到的 `独立程序ZIP/` 只存在于原 ZIP 布局，当前仓库省去重复压缩包；如需原程序 ZIP 或四分卷，可选[原 Release](https://github.com/zzy2670/MA6216-ResNet-Gesture-Control/releases/tag/v2026.11.03)。本目录保留源码、原图、固定分组、权重与实验记录。
>
> 包根 `SHA256SUMS.json` 校验原 ZIP 成员，不能直接校验当前展开仓库树；原机绝对路径和历史校验记录继续保留供追溯。

---

# ResNet手势识别：合并与不合并两种标签方案

本说明位于解压包的`完整源码与数据/Vision_CA/README.md`。需要识别照片时，使用包根目录`程序/`中的独立推理小包；这里保留原图、固定A/B分组、完整训练源码和冻结实验记录，供理解、复核和另开实验。

先读[根交接说明](../../交接说明.md)，需要批注时打开[交接Word](../../交接说明.docx)；方法与两种标签定义查[项目方案v12](../../历史与正式文档/基于ResNet18的手势识别项目方案_v12.docx)，实际结果、错误方向、停止漏识别、原始姿态比较和共同输入速度查[综合实验报告v3](../../历史与正式文档/A与B协议ResNet训练与对比实验报告_v3.docx)。[方案v11](../../历史与正式文档/基于ResNet18的手势识别项目方案_v11.docx)和[报告v2](../../历史与正式文档/A与B协议ResNet训练与对比实验报告_v2.docx)保留不合并阶段的原记录。

## 先选择标签方案，再选择协议和架构

两个方案的输出都只有0—5，区别在于哪些原始姿态可以触发前进1。

| 原始手势 | 不合并 | 合并 |
|---|---|---|
| peace、peace_inverted：V形两种掌背姿态 | 1，前进 | 1，前进 |
| two_up、two_up_inverted：并拢两指的两种姿态 | 0，其他手势 | 1，前进 |
| stop、stop_inverted、palm：三种张掌姿态 | 2，后退 | 2，后退 |
| fist：握拳 | 3，明确停止 | 3，明确停止 |
| like／dislike：拇指向上／向下 | 4，左转／5，右转 | 4，左转／5，右转 |
| 其余手势、没有被接受的有效手势或动作冲突 | 0，无新指令 | 0，无新指令 |

合并扩大了允许前进的姿态范围，也改变了部分照片的正确答案、0类分母和类别分布。相同two_up输入在两方案下的预期不同；总体准确率或宏F1的变化不能直接称为同一任务上的泛化提升。报告分别给出各任务结果，以及共同未变负类上的比较。

A按照片随机划分，允许同一人员跨组；B按人员划分，同一人的全部照片只在一组。两方案沿用同一套固定A/B清单，没有重新分组。ResNet18、34、50是三种分类架构；它们与A/B协议、标签方案是不同的选择维度。

| 标签方案／协议 | 独立程序使用说明 |
|---|---|
| 不合并／A | [ResNet18](../../程序/不合并/A/resnet18/README.md)、[ResNet34](../../程序/不合并/A/resnet34/README.md)、[ResNet50](../../程序/不合并/A/resnet50/README.md) |
| 不合并／B | [ResNet18](../../程序/不合并/B/resnet18/README.md)、[ResNet34](../../程序/不合并/B/resnet34/README.md)、[ResNet50](../../程序/不合并/B/resnet50/README.md) |
| 合并／A | [ResNet18](../../程序/合并/A/resnet18/README.md)、[ResNet34](../../程序/合并/A/resnet34/README.md)、[ResNet50](../../程序/合并/A/resnet50/README.md) |
| 合并／B | [ResNet18](../../程序/合并/B/resnet18/README.md)、[ResNet34](../../程序/合并/B/resnet34/README.md)、[ResNet50](../../程序/合并/B/resnet50/README.md) |

十二个目录各含`predict.py`、`config.json`、分类与检测权重、示例、依赖及校验清单；独立ZIP按同样的两方案和协议保存在包根目录的`独立程序ZIP/`。先固定所选目录，按其配置解释示例和输出。

## 完整照片怎样得到一个数字

SSDLite320 MobileNetV3先在完整RGB照片中检测手框，可以找到多个候选框，也可能漏检或误检。程序按每个保留框裁出手部图，留边、补成正方形并缩放为224×224，再由ResNet计算该手图的六类分数。ResNet可能分错；把多张裁图成批送入分类器，不等于ResNet具有整图检测或多手规则能力。

最终数字由独立Python规则汇总达到分类置信阈值的结果：存在已接受的3时输出3；否则非0有效指令只有一种就输出该指令，多种动作冲突或没有非0有效指令时输出0。两只手都通过阈值后，1＋1→1、1＋2→0、1＋3→3、1＋0→1。没有检测到或未被接受的手不会进入这一步，程序不保证把照片中的所有手都找到并正确分类。

一张照片的公开输出始终只有一个整数。程序内部保留每只手的框、类别、置信度和是否接受；这些明细可由`predict_details`或可选JSON查看，CLI不会逐只手返回多个数字。0表示无新指令，不发起或续期运动，也不等同取消正在执行的运动；3才是明确停止。真正的有效期、取消和小车响应需要控制端按接口约定实现并联调。

## 直接在CPU识别照片

从本README所在的`Vision_CA`文件夹打开PowerShell，以下以不合并B18为例：

```powershell
Set-Location "../../程序/不合并/B/resnet18"
python -m pip install -r requirements.txt
python predict.py "examples/1.jpg" --device cpu
python predict.py "你的完整照片.jpg" --device cpu --json "result.json"
```

换用合并方案时进入`../../程序/合并/B/resnet18`，其他协议或架构同理。模型随推理包提供，不需要训练，也不依赖原电脑的TorchHub缓存；依赖安装可能联网，照片预测不下载模型。示例按固定规则选择，可能预测错误，不保证示例文件名就是程序输出。

成功时stdout只有一行0—5整数。照片不存在、损坏或模型加载失败等错误写入stderr并以非0退出，不能把执行失败当作正常0。`--json`保存每只手的检测框、六类概率、接收状态、整图整数和汇总原因；使用者应同时检查退出码与整数。

批量照片可在所选程序目录创建一个实例，反复调用：

```python
from predict import GesturePredictor

predictor = GesturePredictor(device="cpu", threads=8)
number = predictor.predict("你的完整照片.jpg")
details = predictor.predict_details("你的完整照片.jpg")
```

`predict`返回单个Python整数，`predict_details`返回详细字典，读图等异常向调用方抛出。ResNet分类器本身的接口则是预处理手图张量到六类分数，不能把它单独当作完整照片识别接口。新启动CLI的导入和模型加载时间也不能与报告中模型已加载后的毫秒时延混比。

推理包的`requirements.txt`固定`torch==2.11.0`、`torchvision==0.26.0`、`Pillow==12.1.1`。当前核实的运行环境为Python 3.12.14、PyTorch 2.11.0+cu128、TorchVision 0.26.0+cu128和Pillow 12.1.1，另有训练依赖Optuna 4.9.0、NumPy 2.4.2、pandas 3.0.3；历史训练与检查环境以对应原始记录为准。CPU推理不要求NVIDIA显卡。本机既有环境的运行核验与在新电脑全新安装依赖是两件事；这里未声称已在任意新环境完成安装验证。

## 数据、模型和证据在哪里

```text
Vision_CA/
├─ data/
│  ├─ raw_train_data/hagrid-sample-30k-384p/  原图及原始标注
│  ├─ A_split_verified_20261102/P1_random_image/  固定A三组
│  └─ B_split_verified_20261102/                 固定B三组
├─ pretrained/checkpoints/                      四份官方ImageNet起点
├─ runs/
│  ├─ A_20261102_TPE/                           不合并A结果
│  ├─ B_20261102_TPE/                           不合并B结果
│  └─ MERGED_20261102/{A,B}/{resnet18,resnet34,resnet50}/
│                                               合并结果与逐阶段证据
└─ deployment_src/gesture_predict.py             推理程序代码模板
```

原始数据来自HaGRID的Sample 30K课程副本，共31,833张照片、7,973名人员、38,884个手框。一张照片可能有多只手，照片数与手框数不同。固定A三组照片为25,468／3,185／3,180张；B为25,274／3,457／3,102张。A三组人数允许交叉，B人员互斥；A与B基于同一批原图，两协议之间不要求照片互斥。

不合并版ResNet18的主要结果在各协议运行根目录，ResNet34／ResNet50分别在`compare_resnet34/`和`compare_resnet50/`。合并运行按协议和架构单独保存。查阅`fixed_head_summary.json`、`finetune_summary.json`、`calibration.json`、完整测试预测、权重身份、独审和原始日志，区分准备、训练、验证选型、最终测试与程序验收；当前保存状态不能仅凭某个权重文件存在判断。汇总的准确率、停止错误和速度以v3报告及其绑定的冻结结果为准。

正式报告中的科学图、混淆矩阵、真实错例和任务Grad-CAM保留。个人视频图文学习笔记、讲解媒体、关键帧、转写及会议大录屏不在交接范围。`crops/`派生手图、`features_*.pt`冻结特征和排版渲染缓存不重复收进总包；保留的是原图、固定清单、相关索引、源码和科学证据。四份`pretrained/checkpoints/`官方权重实际包含在包内，与被排除的派生特征缓存不同。

## 阅读训练代码与另开实验

| 代码或记录 | 用途 |
|---|---|
| `prepare_datasets.py`、`prepare_B_datasets.py` | 原图标注整理及实际A/B分组；使用现有固定名单复核时不重新划分 |
| `train_A.py`、`compare_resnets_A.py` | 原六类训练、裁图、冻结特征、微调、验证阈值、完整测试与架构适配 |
| `run_B_pipeline.py`、`run_B_training.py` | 不合并B原运行的阶段组织与监督 |
| `prepare_merged_labels.py` | 合并标签、手图索引、整图真值和已核冻结特征标签的准备逻辑 |
| `run_merged_pipeline.py`、`run_merged_training.py` | 合并六组的固定新头训练、微调、复用检测器、验证冻结与完整测试 |
| `runs/MERGED_20261102/MERGED_PROTOCOL.md` | 合并实验的预先固定流程与参数复用边界 |
| `deployment_src/gesture_predict.py` | 整图检测、裁图、分类、独立规则与整数接口 |

不合并阶段按各自协议、架构进行TPE选型。合并阶段沿用该协议、该架构原先选出的设置，不再TPE；使用新标签训练新分类头，再以官方ImageNet主干和这个新头进行末组微调，没有从旧已微调分类器直接开始。两方案复用相同A/B划分，合并版只复用对应协议的手检测器，不跨协议重用已训练参数。实际完整轮数、选中阶段和验证阈值读运行记录，不把预设上限当作实际完成轮数。

换电脑后，先在本目录检查原图、固定A/B照片和四份官方权重：

```powershell
python portable_ab_project.py check
```

这条命令只读逐文件核对数据和预训练身份，不训练模型，也不验证新模型的完整性能。只有使用推理程序时，不需要重建训练裁图或特征。

若确实需要另开训练，必须使用独立的新输出目录，并根据本机照片位置重新生成裁图、路径索引和冻结特征；合并任务还需要正确的新标签和整图真值。旧索引及日志中的绝对路径用于追溯，不能整体改写科学源JSON后冒称仍是原实验，也不能把不存在的缓存当作已经随包提供。

不合并新A实验入口为`portable_project.py`，不合并新B入口为`portable_B_training.py`，先读各自`--help`；B还可用`--list`查看单阶段顺序。以新的B18裁图准备为例：

```powershell
python portable_B_training.py prepare --out runs/new_B --arch resnet18
```

这只准备一个新阶段，不是照片预测，也不是合并标签训练。原`train_A.py`支持`--split-root`、`--out`和`--pretrained`等实际参数；进一步训练应在新目录中明确数据、标签和官方初始化，保留原冻结结果。

`run_merged_pipeline.py`和`run_merged_training.py`保留本次原运行身份，输出固定在`runs/MERGED_20261102`；`--pretrained-root`可以显式指定官方权重根目录，但不会把它变成独立的新实验。`prepare_merged_labels.py`也按原运行身份读取旧索引和特征。它们可供查阅原算法与监督逻辑，不能直接作为换电脑后一条命令覆盖重跑的入口。另开合并训练时，先在独立副本和新目录完成本机路径、裁图、特征、新标签及真值准备，再确定相应运行入口；不直接重跑已冻结controller，不删除锁或失败记录来重复最终测试。

## 接手复核与小组后续

接手组员先核包根SHA清单，再固定标签方案、A/B和架构，按相应README跑六类、多手、空白和错误路径，保留真实输出与失败照片。示例通过不等于真实无手、光照变化或摄像头连续输入的性能已经验收。

我和组员共同负责ResNet。建议组员主笔ResNet汇报和接收复核，我复核技术；双方结合v3结果选择演示模型，再与另一识别子组确认共同输入、标签、分组和计时口径，与控制组确认整数接口、频率、有效期、0／3及异常处理。各主笔继续落实本部分修改，具体输入、产物和接收方见根交接第8章。这里的识别程序和实验材料不代表ROS、小车控制联调、小组PPT或课程整体展示已经完成。
