# 实验图双语图源

本目录包含15张唯一实验图的中文、英文独立版本，对应正式文档20处图像位置。逐处图注保存在 `manifest.json` 的 `caption_translations`，同源图片在两份文档中的不同编号分别保留。

## 离线重绘

安装 Python 3 与 `numpy`、`Pillow`、`matplotlib` 后，在本目录执行：

```powershell
python render_bilingual_results.py
```

也可定向重绘：

```powershell
python render_bilingual_results.py --only D1-I25 D2-I01
```

绘图只读取本目录的相对路径，不需要原数据集、原项目路径、模型权重、网络或GPU。中文文字默认使用 Windows 常备的 Microsoft YaHei；其他系统需安装支持中文的字体，并在脚本字体列表中指定。PNG已经包含完整字形，SVG中的文字可编辑。`prepare_frozen_inputs.py` 是本次源输入提取记录，不能代替上述离线重绘入口。

## 文件对应

- 各图子目录：`zh.png`、`en.png`、`zh.svg`、`en.svg`、`original_reference.png`，以及该图使用的冻结输入和 `provenance.json`。
- 混淆矩阵：原冻结JSON中的矩阵、样本总数和行归一化尺度，两语言共享同一数据。
- 学习曲线：`frozen_curves.json` 直接保存原日志的完整微调轮次和验证宏平均F1；两语言共享同一组点。
- Grad-CAM：原图科学面板逐像素保留，只有图题、行标、列标和解释文字重排。`row_*_tensors.npz` 保留原存储的归一化224×224 CAM；每个数组的原字节SHA-256记录在 `provenance.json`。不重新前向、反向或归一化，不把颜色当概率或跨层绝对贡献。
- 固定错误案例及裁图：照片、手框、分类结果及裁图像素取自原已发表图；只在原文字空白带替换文字，冻结记录保留样本ID、矩形和数值。
- 四张双手照片：`fixed_photo_1.jpg` 至 `fixed_photo_4.jpg` 与冻结清单的原始照片SHA-256一致，按原比例、坐标、GT框、检测框及检测分数重新排版；“无检测框”文字移至空白边距。检测分数不是分类概率。

`reference_sources` 仅是原始脚本的只读参考副本，保留原作者代码以便溯源；这些入口可能包含训练或推理步骤，**重绘时不要执行**。本目录的离线重绘入口不导入这些脚本。

本目录没有新增训练、推理、准确率或实验样本，也不据图重新估计总体性能。SVG中的照片和CAM是证据栅格，可编辑文字单独保存；定量图的曲线、矩阵、坐标轴和图例可在SVG中编辑。
