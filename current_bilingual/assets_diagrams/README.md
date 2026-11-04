# 十一张项目技术图的中英文版本

`manifest.json` 按正式Word图片ID给出两版PNG、可编辑VSDX、完整中英文图注及SHA256。`zh/` 保留现行正式Word中图片的原始PNG字节和对应中文原生VSDX。`en/` 在同一原生VSDX中翻译全部文字，并在原形状内调整英文换行和字号。所有照片、特征响应图、热图、图形坐标和连线保持原样。

`originals/` 保存未修改的原始VSDX与PNG。VSDX内已经嵌入所需照片和响应图，没有外部关系。生成代码只依赖包内相对路径，不读取原项目数据或模型。图7仍保留早期数据准备记录，其完整图注明确说明这一历史状态。

在安装Microsoft Visio的Windows上，Python环境需要`comtypes`、`Pillow`与`lxml`，并使用系统Arial字体。运行`python generate_diagrams.py`可重新输出图。程序建立独立的`Visio.InvisibleApp`，不使用键盘鼠标，也不接入用户现有Visio会话；仅修改包内生成结果。运行`python validate_diagrams.py`核对原图、数字、原生形状、内嵌图片和文件对应。

本批11张图的原生形状数、坐标和嵌入媒体SHA均已逐项核对，数字token和英文文字完整性检查通过。作者已实际查看所有英文PNG，并修复第一轮换行溢出和窄标签断词；独立审查及最终Word页面检查由文档整合者统一安排。
