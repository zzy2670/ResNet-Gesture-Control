# 完整项目：直接浏览 / Browse the complete project

- [完整项目文件 / Complete project](complete_project/)：源码、数据、实验记录、模型权重和十二个已展开的程序。
- [当前正式文档与中英文图源 / Current documents and bilingual figures](current_bilingual/)：三份正式 Word 及其图源。
- 各程序见 complete_project/程序/；完整项目的原压缩包中另有12份相同程序的 ZIP，此处使用已展开的程序目录。历史清单和说明保留原文，可能仍引用原 ZIP 文件名。
- 仓库较大，可浏览文件或按需下载；完整克隆需下载全部内容。

---

# ResNet手势识别：MA6216项目交接

输入一张RGB照片，先用SSDLite找手，再逐手用ResNet分类，最后汇总为一个0—5整数。包含合并/不合并两种标签方案、A随机照片/B人员互斥两种评价及ResNet18/34/50，共十二个已训练程序。

## 先下载哪个

- **只运行一个模型**：进入[模型与完整项目Release下载页](https://github.com/zzy2670/MA6216-ResNet-Gesture-Control/releases/tag/v2026.11.03)，选例如`resnet18-B-merged.zip`，解压后按照包内README安装依赖，运行`python predict.py examples/1.jpg`。十二个小包各自包含真实权重，不依赖此源码浏览目录。
- **接收全部项目**：先在[原完整项目Release](https://github.com/zzy2670/MA6216-ResNet-Gesture-Control/releases/tag/v2026.11.03)下载`MA6216_ResNet_full_20261103.zip.001`到`.004`四段及`full_project_manifest.json`，放在同一目录。下载本仓库的`restore_full_project.py`，运行`python restore_full_project.py 你的下载目录`，逐段校验并恢复完整ZIP后正常解压。再从[双语文档与图源更新Release](https://github.com/zzy2670/MA6216-ResNet-Gesture-Control/releases/tag/v2026.11.03-bilingual)下载`ResNet_bilingual_documents_and_sources_20261103.zip`并解压，取得当前三份正式文档及配套图源。原四分卷提供完整数据、权重和实验记录，更新ZIP提供当前文档与双语图源，两部分配套使用。
- **只阅读当前文档或使用双语图源**：从[双语文档与图源更新Release](https://github.com/zzy2670/MA6216-ResNet-Gesture-Control/releases/tag/v2026.11.03-bilingual)下载`ResNet_bilingual_documents_and_sources_20261103.zip`。包内`documents/`为三份正式Word；各图源目录提供中英文图、图注、来源清单及重排脚本。
- **在线看代码**：打开[src/Vision_CA](src/Vision_CA)；[configs](configs)是十二包的固定配置。在线浏览树不重复保存原图和大权重，完整源码、数据、逐次训练和原始预测均在上述完整包。

GitHub的“Code → Download ZIP”只下载浏览源码和文档，不包含Release数据与模型。完整项目必须从Release取得。

## 文档阅读顺序

1. [交接说明v2（中英文配图）](docs/交接说明_v2_中英文配图.docx)：项目入口、交接内容和组员后续工作。
2. [综合实验报告v4（中英文配图）](docs/A与B协议ResNet训练与对比实验报告_v4_中英文配图.docx)：全部结果、速度、错误及CAM。
3. [项目方案v13（中英文配图）](docs/基于ResNet18的手势识别项目方案_v13_中英文配图.docx)：方法、数据、架构及实施过程。

当前文档的配图同时提供中文和英文版本，使用同一批冻结实验输出。图源更新包包含原图、可编辑图源和重排方法；其中参考图改编与真实实验结果在图内及图注中区分。历史[交接说明](docs/交接说明.docx)、[综合实验报告v3](docs/A与B协议ResNet训练与对比实验报告_v3.docx)和[项目方案v12](docs/基于ResNet18的手势识别项目方案_v12.docx)继续保留，用于追溯。

## 标签与实际程序

0=无新指令，1=前进，2=后退，3=明确停止，4=左转，5=右转。0不取消正在执行的运动；3取消当前运动。`stop`原始标签是手形名称，本项目将张掌映射为2，握拳才映射为3。

不合并版将peace与peace_inverted设为1；合并版再加入two_up与two_up_inverted。两个版本都保留stop、stop_inverted、palm合为2。未选手势、闲置手和不确定结果可输出0。

每个手框通过阈值后，3优先；其余唯一有效非零命令则输出该值，运动命令冲突输出0。内部每手结果可存JSON，正常CLI只打印一个整数。这是额外Python汇总规则，不是ResNet自身的能力；检测可能漏手，手图分类准确不等于整图命令准确。

## 已完成的真实测试

|标签方案|协议|模型|手图分类准确率|整图命令准确率|整图CPU均值/ms|
|---|---|---|---:|---:|---:|
|不合并|A|resnet18|99.34%|93.62%|32.69|
|不合并|A|resnet34|99.18%|93.14%|43.39|
|不合并|A|resnet50|99.44%|94.34%|50.65|
|不合并|B|resnet18|99.15%|93.07%|32.89|
|不合并|B|resnet34|99.12%|92.68%|43.30|
|不合并|B|resnet50|99.02%|93.33%|49.62|
|合并|A|resnet18|99.06%|91.95%|32.90|
|合并|A|resnet34|98.90%|92.17%|43.88|
|合并|A|resnet50|99.36%|92.89%|49.98|
|合并|B|resnet18|99.07%|91.84%|33.07|
|合并|B|resnet34|99.07%|91.81%|43.25|
|合并|B|resnet50|98.94%|92.42%|50.55|

A测试3180张照片，B测试3102张。手图分类使用人工框，整图指标包含实际检测与拒识/汇总；CPU为同一机器、8线程、固定相同输入下的均值，不代表所有电脑。新旧标签定义改变了负类分母，应结合报告里的相同原始负类比较，不能仅靠两行准确率判断合并更好。

完整ZIP为6,188,474,282字节，96,986个成员，SHA256 `4448838cbc4e8a4a5cf4199a20ad593db996e1a5cf1d6208b9a24129d60a2a96`。已独立检查全部成员SHA/CRC、十二内ZIP、正式文档和固定数据身份；原图/A/B各31,833张。个人学习笔记、音视频和转写未纳入，原始实验失败与恢复、科研CAM均保留。

## 数据与使用范围

采用课程指定HaGRID Sample30K，见[来源及原许可](THIRD_PARTY.md)。本仓库用于课程组内交接，保持私有；组员需要仓库访问权限。没有为第三方数据、课程文件和作者材料另行授予统一开源许可。

十二推理小包可按各自README迁移。历史训练控制器含原机索引；跨机重训须从完整包原图与固定名单重建索引，并写新运行目录，不覆盖原成果。原历史文档的部分本机路径保留，详见交接说明。识别交付不等于已完成机器人/ROS集成或整门课程验收。
