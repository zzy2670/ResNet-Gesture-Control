# 合并标签新训练模型分层Grad-CAM准备协议

当前仅工具准备；不代表合并六模型训练/测试完成，不代表CPU测速完成，不代表新CAM已生成或通过独审。此文件及同目录make_merged_trained_gradcam.py由CAM作者独占写入；旧A/B脚本、模型、图和所有输出只读。

## 来源与固定范围

原make_A_trained_gradcam.py与make_B_trained_gradcam.py是已验证公式与呈现来源，两脚本SHA也绑定到新证据。新工具只适配新runs/MERGED_20261102/{A,B}/resnet18。每协议一张6行×5列大图：分类输入、layer4（7×7）、layer1（56×56）、layer2（28×28）、layer3（14×14）。每层最后一个残差块的输出保存activation及gradient。目标固定真实类别logit，错例也保留，不切换预测类或更好看类别；不新增性能分数，不修改test阈值或分母。

真实公式保持为gradient的空间平均作为通道权重，ReLU(sum(weight×activation))，双线性上采样224×224且align_corners=False，再逐样本逐层min-max；常数图为零。原输入RGB与turbo颜色按0.58/0.42叠加。独立归一化意味着不能跨层、跨样本比较绝对颜色强度；颜色不是准确率，区域关联不是物理因果。

## 固定样例规则

先查原A/B trained_gradcam/metadata.json对应真实类别的sample key(image_id, hand_index)，在新crops_test.json中保留同一个身份。若其标签已变、不能继续担当旧行，就按不可变key排序取新该类首个合法样本，并记录旧key、original_label与替换原因。不读取预测成功情况、置信度或CAM来挑图。新旧测试key全集必须相同，像素路径、source_path、GT bbox_xyxy、crop_rectangle、raw_label和user_id必须原样相同。

新class0严格排除peace、peace_inverted、two_up、two_up_inverted，尤其不能把原two_up=0热图改标。新class1允许这四种姿态；若固定规则选中two_up，必须加载真实新训练checkpoint并真实前向反向，不能重标旧图。默认优先复用旧固定class1 peace身份，因此不保证示例一定含two_up；四raw子群的性能仍来自完整已审测试，不用这六张解释图估计。

--plan-samples仅使用标准库读取索引和旧metadata，打印固定样例；不打开图片、不加载权重、不推理。root可以将其标准输出存为预先固定选择回执。所有六类即使预测错误或热图分散也必须保留，不自动换图。

## 原图、GT与检测裁图的关系

主矩阵使用人工GT框产生的原224×224分类输入，属于人工框条件解释，不能称为完整检测流程解释。metadata逐行保留原source记录及原图/GT框/crop_rectangle；另带同照片冻结whole_photo prediction的全部hands（检测框、检测score、实际分类概率等原字段），并保存冻结oracle预测以独立核对新CPU重推理。

人工GT框裁图和检测器裁图不是同一输入。附带冻结检测框能追踪错误链条，不能把人工框CAM平移后冒称检测裁图的CAM。未另安排真实检测裁图CAM，因此本工具不加载检测器、不重推理整图。解释稿应查看原照片及GT手部位置，说明热点落在手指、掌面、边缘还是背景；不清楚就如实写不清楚。定位失败、分类失败、阈值拒识、规则聚合分别从冻结证据判断，不能只看热图断言。

## 正式运行门禁

默认、--help、--schema均不加载机器学习库或模型。正式--run-cpu先复用已独审benchmark_label_variants_cpu.scientific_readiness，要求all_training_evaluation_complete.json=PASS、aggregate_review.json=PASS_MERGED_SIX_MODEL_AUDIT_WITH_LIMITATIONS、六个逐模型独审及其全部绑定身份一致；缺失即失败。

随后要求label_variants_cpu_benchmark.json=complete及真正速度独审，独审必须通过data_sha256或bound_file_sha256绑定当前速度文件。尚无最终速度独审时不伪造兼容PASS。root提供UTF-8 JSON许可：purpose=merged_gradcam_cpu，confirmed_by=root，cpu_window_confirmed=true，expires_at为带时区ISO8601，speed_review_path为真实独审绝对路径，speed_review_status为该独审准确PASS状态，bound_file_sha256至少绑定当前速度JSON及独审文件。许可来自root调度，不是脚本自动推定的用户授权。

真正调用示例（仅说明，当前不执行）：

`python runs/MERGED_20261102/make_merged_trained_gradcam.py --run-cpu --protocol A --threads 8 --permit <root提供的许可绝对路径>`

完成A后由root安排B窗口，--protocol B；8线程默认，允许root在1至8内调整。仅CPU，CUDA_VISIBLE_DEVICES置空，不调用torch.cuda，不训练、不加载检测器、不联网；严格加载对应真实新checkpoint。所用解释器和依赖由root沿用已有验证环境，本轮AST/help不构成依赖运行通过。

## 输出与独审接口

新输出只在{A,B}/resnet18/trained_gradcam；已存在目录拒绝覆盖与自动重跑。失败产生的部分文件保留，重跑需root审核原因并另行安排，不删除后假装第一次成功。只有最后metadata.json提交才形成COMPLETE_PENDING_INDEPENDENT_CAM_REVIEW；这不是独审PASS。

每行row_N_tensors.npz包括input_normalized_tensor[3,224,224]、input_rgb[224,224,3]、logits[6]，每层的activation[C,H,W]、gradient同形、channel_weights[C]、native_cam[H,W]、upsampled_224_cam[224,224]、normalized_224_cam[224,224]。原activation/gradient均保存真实数值，不用可视化图反推张量。

metadata.json含六个唯一真实类别、样例和旧新标签身份、真实CPU概率及与冻结oracle概率的最大绝对差、保存整图预测、输入tensor SHA、双重门禁与全部绑定文件SHA、公式、归一化、输出SHA。冻结概率可能因CPU/GPU数值差有小偏差，独审须实际量化并判断，不能自动宣称完全相同。

独审逐层独立从gradient求通道平均，重建ReLU加权图、上采样及每图归一化，核原数值finite、尺寸、零/常数图处理、目标真实logit和六个唯一类；再核RGB输入/tensor预处理、样例身份与完整冻结预测、概率差异、矩阵每格与对应npz、原图手部关系、文字可读及限制。不能只审layer4，也不能只因四层图像相似宣称正确。图片和npz先编码独占提交，完成metadata在精确字节形成后复核实时许可及全部来源身份再独占提交；旧结果始终不写。

成功metadata字节物化及全部来源、七份输出SHA复核完成后，程序紧邻提交重读许可字节、核其SHA仍与门禁绑定一致，再核purpose/root/CPU确认和当前带时区expiry；此处不重复耗时大文件哈希，以免许可在长核验期间过期却留下成功标记。

此准备任务结束后释放作者slot，不等待六模型，不实际执行CAM，不生成缺失数据Word。后续root在所有科学与测速门禁满足后负责调度和独立结果审核。
