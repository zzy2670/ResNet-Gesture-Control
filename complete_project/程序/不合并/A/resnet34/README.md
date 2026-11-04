# ResNet34：输入照片，输出0—5

本文件夹包含已训练的ResNet34分类器、共用的手部检测器和冻结阈值。直接输入完整照片，无需先手动裁出手，也不需要原训练数据或原项目目录。

## 快速使用

解压整个文件夹，保留`predict.py`、`config.json`和`weights`的相对位置。使用Python 3.12，在该文件夹打开终端。

```powershell
python -m pip install -r requirements.txt
python predict.py "examples/1.jpg"
python predict.py "你的照片.jpg"
```

终端标准输出只有一个整数，例如`1`；错误写入标准错误并返回非零退出码。权重已放在包中，预测不会联网下载模型。依赖第一次安装需要网络；本机已有相同依赖时无需再次安装。程序默认使用CPU，CUDA环境可显式加`--device cuda`。

## 输出对应关系

| 数字 | 手势 | 约定的控制含义 |
|---|---|---|
| 0 | 其他手势、无被接受的有效手势或有效动作冲突 | 无有效动作指令 |
| 1 | peace、peace_inverted：V形手势 | 前进 |
| 2 | stop、stop_inverted、palm：张开手掌 | 后退 |
| 3 | fist：握拳 | 停止 |
| 4 | like：拇指向上 | 左转 |
| 5 | dislike：拇指向下 | 右转 |

程序仅输出识别结果，不连接机器人。多手时，已接受的握拳优先输出3；否则仅有一种有效动作就输出该数字，不同有效动作冲突则输出0。0类手不会抵消另一只手的单一有效指令。没有通过检测与分类门槛的手也输出0。

## 查看框与概率

```powershell
python predict.py "你的照片.jpg" --json "result.json"
```

JSON保留各手检测框、检测分数、六类概率、接受状态和最终整数。`config.json`中的检测阈值为0.3、分类阈值为0.5，均来自验证集，未按新图片调整。

## 在自己的Python程序里使用

```python
from predict import GesturePredictor

# 加载一次模型，多张图片重复调用，避免每次重新读权重。
model = GesturePredictor()
number = model.predict("你的照片.jpg")
print(number)
```

首次启动包含导入依赖、校验和加载模型，需额外时间；报告中的毫秒耗时指模型已加载后的照片处理时间。样例来自验证集固定规则选取，可能识别错误；`samples.json`记录真实类别，不是程序写死的返回值。空白图只是接口检查，不代表已完成真实无手场景的性能测试。

## 文件结构

- `predict.py`：命令行入口和可复用的`GesturePredictor`。
- `config.json`：当前模型架构、固定阈值、类别映射与权重SHA256。
- `weights/classifier.pt`：本架构的六类分类权重。
- `weights/detector.pt`：已训练的离线手部检测权重。
- `examples`、`samples.json`：完整照片示例和来源说明。
- `requirements.txt`：本轮验证过的核心依赖版本。
- `SHA256SUMS.json`：各文件的完整性清单。

原始训练、验证、逐样本测试、故障修复与独立核查记录见统一交付文件夹的`训练记录`。本包只用于推理；A协议允许同一人员出现在不同分组，不能将该成绩当成人员互斥泛化或机器人实际控制效果。
