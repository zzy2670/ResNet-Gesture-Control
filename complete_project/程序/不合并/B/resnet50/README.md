# B协议ResNet50：输入完整照片，输出0—5

本包使用按人员互斥划分的B训练模型，与原A程序分开保存。三种架构的B检测器相同，分类器分别训练；检测器和分类器均没有继承A的微调权重。

## 使用步骤

1. 保留整个文件夹，在Python 3.12环境安装`requirements.txt`。
2. 执行`python predict.py "examples/1.jpg"`或将路径换成自己的完整照片。
3. 标准输出为一个整数。需保留框和概率时加`--json result.json`。

依赖安装可能需要网络；模型文件已经包含，推理不会联网下载权重。默认CPU，可显式加`--device cuda`。读图或加载失败写标准错误并返回非零退出码，不伪装成0。

| 数字 | 手势 | 控制约定 |
|---|---|---|
| 0 | 其他、无有效动作或动作冲突 | 不发起或续期运动 |
| 1 | peace、peace_inverted | 前进 |
| 2 | stop、stop_inverted、palm | 后退 |
| 3 | fist | 明确停止当前运动 |
| 4 | like | 左转 |
| 5 | dislike | 右转 |

已接受的握拳优先输出3；否则非零有效动作只有一种就输出该数字，不同动作冲突输出0。0类手不抵消另一只手的单一有效指令。程序仅识别图片，不会控制机器人。

## 多张照片

```python
from predict import GesturePredictor

# 模型只加载一次，后续重复调用。
model = GesturePredictor()
number = model.predict("自己的照片.jpg")
print(number)
```

`config.json`保存架构、类别映射、预处理、检测阈值0.3、分类阈值0.5和权重SHA256。阈值仅由B验证集确定，不能按演示图片修改后仍引用原测试分数。

`weights`是两个冻结权重；`predict.py`是离线架构与CLI/API；`samples.json`记录验证样例及真实标签；`SHA256SUMS.json`用于完整性核查。示例按固定ID选取，可能预测错误；空白图只是合成接口检查，不等于真实无手场景准确率。

报告毫秒值是模型加载后的耗时，不包含每次启动Python和读权重。B测试覆盖本地HaGRID样本中的未见人员，不等于实机机器人、所有背景或新设备已通过验收。
