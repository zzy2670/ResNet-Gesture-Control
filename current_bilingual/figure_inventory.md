# 当前正式文档双语图与图源盘点

本轮只读核对三份正式Word，共44处可见图片，39个不同SHA256。当前正文尚未形成独立的中文版与英文版图对。9张无文字照片、2张外部英文原图、28张项目自制中文技术图。图内文字翻译和两版正文嵌入均须完成，翻译图注不能计为完成。

方案v12为29图，实验报告v3为15图，完整交接说明为0图。所有44处图片均找到项目内字节完全一致的源图片；原始嵌入图片已按SHA提取至embedded_originals。

13张方案自制图有现存可编辑VSDX，读取原生文字可追溯，但本轮未重新导出，导出像素身份待后续验证。15张不同实验图可用现有保存结果重新排版；4套正式Grad-CAM均有6份张量与metadata，不需要再运行模型。

两张电缆来源参考图（方案图10、图12）必须保留原英文图及来源，中文版只能标明是依据原图翻译的适配图；不能伪造外部作者的可编辑原图。无文字照片的中英图可复用同一照片像素，两语言版本分别保留在正文。

## 逐图状态

|图ID|现行图题|图内语言|同源制作路径|首选源|
|---|---|---|---|---|
|D1-I01|手势编码表内照片 peace|无图内文字，真实照片|原照片双语配套|视频笔记_QQ20261101-180117/素材/v4_feature_input_hand.png|
|D1-I02|手势编码表内照片 peace_inverted|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/peace_inverted_hand.png|
|D1-I03|手势编码表内照片 stop|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/stop_hand.png|
|D1-I04|手势编码表内照片 stop_inverted|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/stop_inverted_hand.png|
|D1-I05|手势编码表内照片 palm|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/palm_hand.png|
|D1-I06|手势编码表内照片 fist|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/fist_hand.png|
|D1-I07|手势编码表内照片 like|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/like_hand.png|
|D1-I08|手势编码表内照片 dislike|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/dislike_hand.png|
|D1-I09|手势编码表内照片 no_gesture|无图内文字，真实照片|原照片双语配套|ResNet项目完整交接_20261102/历史文档/数据集说明/class_examples/no_gesture_hand.png|
|D1-I10|图2a 普通CNN与残差CNN的连接方式对照。残差块把主分支F(x)与旁路x逐位置相加；5＋1＝6是单个特征位置的教学算例。残差连接帮助深层网络训练，不代表本项目准确率已提高。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v7_fig2a_residual_compare.png|
|D1-I11|图1 项目总体流程。完整照片→SSDLite320自动找手→按框裁图并预处理为224×224→ResNet18六类手势分类→可靠性与多手汇总规则→整图0—5整数。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v8_overall_pipeline_final.png|
|D1-I12|图2 ResNet18内部结构。初始卷积和池化后依次通过四组残差块，再经平均池化得到512维特征；本项目采用新的512→6分类层。下部展开残差块中的计算分支、旁路和相加位置。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/resnet_plan_v2_resnet18_model.png|
|D1-I13|图3 ResNet18网络结构与特征计算示例。上部为完整网络结构，下部对应展开四阶段特征响应、局部卷积、残差相加及全局平均池化。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_architecture.png|
|D1-I14|图4 PyTorch在本项目中的训练与预测过程。训练误差用于调整可学习参数；训练完成后保存参数，预测阶段加载参数并计算输出。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_fig_4_pytorch.png|
|D1-I15|图5 TorchVision网络代码、ImageNet预训练参数与六分类改造。网络结构与参数分别提供计算路径和训练起点，新分类层通过本项目手势数据学习。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_fig_5_torchvision.png|
|D1-I16|图6 HaGRID数据来源及标注用途。照片、手框、手势标签与人员编号分别支持完整图像输入、手定位、手势分类和人员分组。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_fig_6_hagrid.png|
|D1-I17|图7 早期Vision_CA数据准备示意。图中“仅有summary统计”的状态属于当时的旧输出；本次实际使用的新A、B三组清单均已生成并核验，数量与位置见第1.2节。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_fig_7_visionca.png|
|D1-I18|图8 自动手定位与手势分类的衔接。SSDLite320使用完整照片学习手框，ResNet18使用裁手图学习手势，整图结果经规则汇总为0—5。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_fig_8_ssdlite.png|
|D1-I19|图9 开源资源与本项目实现的总体关系。数据、整理代码、网络结构及通用权重共同支持本项目训练，训练后的定位与分类模型构成完整预测程序。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v5_fig_9_summary.png|
|D1-I20|图10 电缆项目的解释路径原图参考。原图来自本地电缆项目报告图3.9，展示前向推理及目标层解释计算；图中的逐位置特征×梯度后跨通道平均对应原项目实现。本图只作为计算流程参考，不是本项目手势实验结果。|英文原图|外部原图+忠实翻译适配|视频笔记_QQ20261101-180117/素材/v7_cable_study_figure_3_9.png|
|D1-I21|图11 本项目ResNet18的标准Grad-CAM计算机制示意。前向取得目标类别分数，解释路径先对梯度作空间平均，再用所得通道权重组合特征，经过ReLU后，先上采样至输入尺寸，再逐图归一化并叠加到输入。本图说明计算机制，不给出六类手势识别成绩。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v10_our_gradcam_mechanism.png|
|D1-I22|图12 电缆论文分类别分层总览原图参考。原图为本地电缆论文图4.14，七行对应七类测量状态，四列依次为原始STFT、block3、block1、block2。本图作为类别与模块两个维度的呈现参考，不是手势实验结果。|英文原图|外部原图+忠实翻译适配|视频笔记_QQ20261101-180117/素材/v7_cable_study_v42_figure_4_9.png|
|D1-I23|图13 训练前制作的六类手图分层布局示例。每行按来源标签0—5组织，列为输入、layer4主解释、layer1、layer2、layer3。热图由通用ImageNet ResNet18实际计算，每行标注其真实ImageNet目标；保留本图用于说明布局，不将它改称本项目六类识别结果。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v9_cam_matrix.png|
|D1-I24|图14 原图、响应与手指局部的对应观察示例。方框和放大内容来自本图同一输入的同一位置，可查看伸直手指的轮廓及两指间空隙。此独立样本以通用ImageNet模型的nail（钉子，索引677）为目标，不表示已正确识别V形手势；正式局部分析按实际需要补充。|中文主导，含模型名/公式/英文术语；无独立英文版|原生VSDX复制翻译|视频笔记_QQ20261101-180117/素材/v7_fig12_local_evidence.png|
|D1-I25|图15 A协议已训练ResNet18六类分层Grad-CAM。真实测试手图与固定任务权重，各图独立归一化；该图不表示跨层响应强弱或总体性能。|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/trained_gradcam/trained_six_class_matrix.png|
|D1-I26|图16 B协议已训练ResNet18六类分层Grad-CAM。解释真实类logit，各图独立归一化；颜色不等于概率，不能比较跨层绝对强弱。|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/trained_gradcam/trained_six_class_matrix.png|
|D1-I27|图17 B测试照片的人工框分类与自动定位输入对照。上例右图仅为最高检测分数框，未声称一定与左侧真框匹配；下例没有通过阈值的自动框。|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/report_figures/actual_crop_comparison.png|
|D1-I28|图18 合并A ResNet18六类分层Grad-CAM。人工GT框、真实类logit，各图独立归一化；不能据颜色深浅跨层比较绝对贡献。|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/MERGED_20261102/A/resnet18/trained_gradcam/trained_six_class_matrix.png|
|D1-I29|图19 合并B ResNet18六类分层Grad-CAM。人工GT框、真实类logit，各图独立归一化；不能据颜色深浅跨层比较绝对贡献。|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/MERGED_20261102/B/resnet18/trained_gradcam/trained_six_class_matrix.png|
|D2-I01|图1 ResNet18人工框手图测试混淆矩阵|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/report_figures/ResNet18_crop_confusion.png|
|D2-I02|图2 ResNet18完整照片测试混淆矩阵|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/report_figures/ResNet18_whole_confusion.png|
|D2-I03|图3 ResNet18四类固定整图错误案例|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/report_figures/whole_photo_error_examples.png|
|D2-I04|图4 同一照片人工框与自动框实际裁图对照|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/report_figures/actual_crop_comparison.png|
|D2-I05|图5 已训练ResNet18六类手图分层Grad-CAM|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/trained_gradcam/trained_six_class_matrix.png|
|D2-I06|图6 分类网络逐轮验证宏平均F1学习曲线|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/A_20261102_TPE/report_figures/classifier_learning_curves.png|
|D2-I07|图7 B ResNet18人工框测试混淆矩阵|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/report_figures/ResNet18_crop_confusion.png|
|D2-I08|图8 B ResNet18整图测试混淆矩阵|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/report_figures/ResNet18_whole_confusion.png|
|D2-I09|图9 B已训练ResNet18六类分层Grad-CAM|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/trained_gradcam/trained_six_class_matrix.png|
|D2-I10|图10 B ResNet18四类固定整图错误案例|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/report_figures/whole_photo_error_examples.png|
|D2-I11|图11 B同一照片人工框与自动定位实际输入对照|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/report_figures/actual_crop_comparison.png|
|D2-I12|图12 A/B三架构完整微调验证曲线|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/B_20261102_TPE/report_figures/AB_classifier_learning_curves.png|
|D2-I13|图13 合并A已训练ResNet18六类人工框手图分层Grad-CAM|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/MERGED_20261102/A/resnet18/trained_gradcam/trained_six_class_matrix.png|
|D2-I14|图14 合并B已训练ResNet18六类人工框手图分层Grad-CAM|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/MERGED_20261102/B/resnet18/trained_gradcam/trained_six_class_matrix.png|
|D2-I15|图15 不合并B18四张双手照片的人工标注与冻结检测对照|中文主导，含模型名/类别代码/英文术语；无独立英文版|已保存证据只排版|Vision_CA/runs/MERGED_20261102/detection_spotcheck_20261103/四张双手照片_人工标注与检测对照.png|

## 去重与分工依赖

正文有5张重复使用图：A正式Grad-CAM、B正式Grad-CAM、B裁图对照、合并A Grad-CAM、合并B Grad-CAM。每一对语言图只生成一次，供两份正式Word复用。

可并行进行方案技术图翻译和实验结果图排版；各自产出独占目录。两组完成后由单一整合者写新版Word并打ZIP，独立审者核图内所有文字、数值/科学关系、源文件与嵌入对应和实际Word页面。历史Word、冻结科学文件与当前上传旧ZIP均不修改。

详细逐图关系、SHA、源路径、原生文本、生成脚本、数值证据、重复组及限制见figure_inventory.json。

## 本轮检查边界

最终候选搜索排除HaGRID、data、crops、train、val、test、images、dataset、全部渲染/预览/排版目录。初版因crops不叫data曾进入该目录并读取候选图进行SHA比较，随即停止并扩大剪枝；该意外目录信息未用于本盘点结论。

本轮未训练、未推理、未改原Word或原生图，未碰远端GitHub，也未操作旧ZIP上传。只检查图片内容与包结构，不声称已做Word完整页面渲染。
