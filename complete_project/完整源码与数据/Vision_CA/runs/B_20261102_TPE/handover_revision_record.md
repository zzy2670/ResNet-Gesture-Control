# A与B根交接修订记录

root另行移交新版根`交接说明.md`、`交接说明.docx`、`README.md`唯一写入权。旧根MD八章及旧Word七章已完整读，确认旧Word没有第8章建议分工；旧文件均不覆盖。

内部初稿`handover_working.md`已形成8节：通俗项目与0—5；来源、A/B实际三计数和真实训练；完成清单及结果；新版与历史文档阅读顺序；CLI/API；完整ZIP布局；课程和控制衔接边界；本人和组员的建议分工及对应输入、产物、接收方。根目录本地暂存不含数据是stream布局的正常状态，最终相对链接按freeze成员集合验证，不把暂存形态写给接手读者。

保留方案／报告中的正式科学图与CAM，删除旧交接的图文学习笔记阅读整节，不分发讲解视频和会议大录屏。历史方案v1—v10与A报告v1保留原文，用新版说明其当时状态；不重复派已经完成的A/B训练从头执行。

已核真实CLI是照片位置参数，`python predict.py 照片路径`，成功仅整数、错误stderr非0。独立程序把部署源码命名为predict.py，API示例应为`from predict import GesturePredictor`；六类映射内置config.json，并无mapping.json。当前portable_project.py check只覆盖原图／A／四预训练，B不得误称由此检查。

`build_AB_handover_docs.py`已准备，复用旧工具的Markdown行内链接解析，重新设置A4、常规中文字号、黑色标题、显式CJK字体、轻灰表格边界、重复表头和行不拆分。正式生成要求已审核事实JSON的PASS、六程序已核验标志和逐文件SHA一致；新根三文件均在最终证据后生成。生成后由root安排独立内容与逐页视觉审查，链接再按最终freeze成员集合核对。

当前依赖：B34/50最终结果与统一CPU；图计算独审；六程序验收；准确的AB报告文件名；最终成员清单。当前仅内部MD，不冒称正式交接完成。


## 2026-11-02 最终证据与作者渲染自查

正式交接已依据三模型科学独审、六程序readiness与共同输入CPU独审PASS生成。方案v11为28页8表27图，原24图SHA及次数保全；交接为7页8章，根MD与Word来自同一最终Markdown。共同输入CPU采用全部三轮300次，背景负载7.4%—43%与B18第三轮50.54ms保留，原准确率和原速度记录不改。B50实际11完整轮/best7，恢复后的1手标签变化、全部概率差异与测试访问均披露。

全部35张最终PNG已由作者逐页view自查，无裁切、重叠、缺字或图片丢失；精确SHA、分页、渲染目录及全页清单见author_document_render_review.json。非作者正在独审，正式文件此后不得改动；需要修订时先通知审者并重渲染受影响内容。没有把作者自查当成独审PASS。


### 独审反馈实改与最新保存态

已移除方案继承的15个个人媒体绝对file:点击关系，嵌入原24图及可见文字保持；剩7个公开网页链接。图15改为各图独立归一化。最终重渲染为project_plan_v11_final_render2和handover_AB_final_render2；方案仅第24页PNG字节变化，已重新view；其余27页及交接全部7页与已逐页审读版本PNG字节一致。最新SHA已同步author_document_render_review.json；等待非作者关闭本批反馈。


### 完整包缓存说明与真实成员规则对齐

独审核到build_ab_project_bundle.py的filter_cache会排除crops及features_*.pt，已把交接第6节改为保留原图、固定A/B照片、索引和重建代码；派生裁图与特征缓存换机重建。方案及根README未改。最新交接render3仍7页8章，1—4页PNG字节一致，5—7页已重新view。最新SHA见作者渲染审查JSON。
