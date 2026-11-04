# 合并标签 B协议 ResNet50

输入完整照片，输出0—5。1包含peace、peace_inverted、two_up、two_up_inverted，表示V形或并拢两指前进。此包使用重新训练的合并分类器，阈值由相同协议的新验证结果固定。

安装`requirements.txt`后执行`python predict.py "examples/two_up.jpg"`，或换成自己的照片。默认CPU；需要框与概率可加`--json result.json`。模型已包含，推理不下载权重。标准输出只有整数；读图或加载失败写stderr并非零退出。

0为无有效动作，1前进，2张掌后退，3握拳停止，4拇指向上左转，5拇指向下右转。多手握拳优先；其余不同非零动作冲突输出0。程序只识别图片，不连接机器人。A允许同人跨组；B人员互斥，两者的分数不可直接混为部署保证。

`config.json`保存协议、合并版本、固定分组SHA、双权重SHA和新val阈值。示例按验证ID固定选择，可能识别错误；它们不构成新增benchmark。合成空白图仅用于接口测试。

连续图片可`from predict import GesturePredictor`，加载一次`model=GesturePredictor()`后调用`model.predict("图片.jpg")`。计时应区分加载和稳定推理。
