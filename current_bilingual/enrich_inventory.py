"""只读核对图内文字、原生源和已有数值证据，补全图源盘点。"""
from pathlib import Path
from zipfile import ZipFile
from hashlib import sha256
from xml.etree import ElementTree as ET
import json, re

O=Path(__file__).resolve().parent
P=O.parent
V=P/'Vision_CA'
S=P/'视频笔记_QQ20261101-180117/素材'
d=json.loads((O/'figure_inventory.json').read_text(encoding='utf-8'))
native_map={10:'v7_fig2a_residual_compare.vsdx',11:'v8_overall_pipeline_final.vsdx',12:'resnet_plan_v2_resnet18_model.vsdx',13:'v5_architecture_r4.vsdx',14:'v5_fig_4_pytorch_r2.vsdx',15:'v5_fig_5_torchvision_r2.vsdx',16:'v5_fig_6_hagrid_r2.vsdx',17:'v5_fig_7_visionca_final_r2.vsdx',18:'v5_fig_8_ssdlite_r2.vsdx',19:'v5_fig_9_summary_r2.vsdx',21:'v10_our_gradcam_mechanism.vsdx',23:'v9_cam_matrix.vsdx',24:'v7_fig12_local_evidence.vsdx'}
gen_map={10:'v7_residual_compare_create.ps1',11:'v8_pipeline_create.ps1',12:'resnet_plan_v2_create_model.ps1',13:'v5_architecture_create.ps1',14:'v5_chapter5_figures_create.ps1',15:'v5_chapter5_figures_create.ps1',16:'v5_chapter5_figures_create.ps1',17:'v5_fig7_regenerate.ps1',18:'v5_chapter5_figures_create.ps1',19:'v5_chapter5_figures_create.ps1',21:'v10_gradcam_mechanism_create.ps1',23:'v9_cam_matrix.py',24:'v7_cam_figures_create.ps1'}
def binding(p):
    return {'path':str(p),'exists':p.is_file(),'sha256':sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
for r in d['figures']:
    rid=r['id']; n=r['sequence']; r['paired_zh_en_in_current_body']=False
    # 优先采用现行运行目录或原素材，避免把归档副本当作当前唯一源。
    sources=[Path(x) for x in r['exact_source_files']]
    preferred=next((p for p in sources if p.is_relative_to(V)),None) or next((p for p in sources if p.is_relative_to(S)),None) or sources[0]
    r['preferred_exact_source']=binding(preferred)
    r['native_sources']=[];r['recommended_generators']=[];r['evidence_sources']=[]
    r['scientific_change_allowed']=False
    if rid.startswith('D1-') and n<=9:
        r['caption']='手势编码表内照片 '+['peace','peace_inverted','stop','stop_inverted','palm','fist','like','dislike','no_gesture'][n-1]
        r['image_text_language']='无图内文字，真实照片'
        r['bilingual_route']='同一照片原像素在正文中保留中文版与英文版两处；以两语言的图题/说明区分，勿凭空向照片绘制内容。'
        r['regeneration_without_model']=True
    elif rid in {'D1-I20','D1-I22'}:
        r['image_text_language']='英文原图'
        r['bilingual_route']='保留英文原图字节与来源；中文版作忠实翻译适配并显式标明依据原图翻译，不能冒称项目自制结果或替作者补造原生图。'
        r['external_original']=True
        r['regeneration_without_model']='可作翻译适配；原图无已核原生源'
    elif rid.startswith('D1-') and n in native_map:
        r['image_text_language']='中文主导，含模型名/公式/英文术语；无独立英文版'
        p=S/native_map[n];b=binding(p)
        # 读取原生文本，单独记录结构可编辑性；尚未重新导出，不能声称导出像素身份已核。
        with ZipFile(p) as z:
            parts=[x for x in z.namelist() if re.match(r'visio/pages/page\d+\.xml$',x)]
            texts=[]
            for part in parts:
                root=ET.fromstring(z.read(part))
                texts.extend(''.join(e.itertext()) for e in root.iter() if e.tag.endswith('}Text'))
            b.update(native_text_count=len(texts),native_chinese_count=sum(bool(re.search('[\u4e00-\u9fff]',t)) for t in texts),native_texts=texts)
        r['native_sources'].append(b)
        r['recommended_generators']=[binding(S/gen_map[n])]
        r['bilingual_route']='复制原生源为zh/en两版，逐文本替换并保持科学关系与原图内嵌照片/热图像素；导出后核同源对应及所有标签。'
        r['regeneration_without_model']=True
        r['native_export_identity']='同名/已读原生文字与历史脚本对应；本轮未执行Visio导出，不声称像素身份已核'
        if n==13:
            r['evidence_sources']=[binding(S/'v4_feature_tensors.npz'),binding(S/'v4_feature_metadata.json')]
        if n in {21,24}:
            r['evidence_sources']=[binding(S/'v7_cam_tensors.npz'),binding(S/'v7_cam_metadata.json')]
        if n==23:
            r['evidence_sources']=[binding(S/'v9_cam_matrix_metadata.json')]+[binding(S/f'v9_cam_row{i}_tensors.npz') for i in range(6)]
            r['recommended_generators'].append(binding(S/'v9_cam_matrix_visio.ps1'))
    else:
        r['image_text_language']='中文主导，含模型名/类别代码/英文术语；无独立英文版'
        r['regeneration_without_model']=True
        r['bilingual_route']='只用已有数值、照片、预测或保存的热图作zh/en排版；独立语言字典驱动同一绘图实现；不得执行原训练/解释推理入口。'
        parent=preferred.parent
        if parent.name=='trained_gradcam':
            r['evidence_sources']=[binding(parent/'metadata.json')]+[binding(parent/f'row_{i}_tensors.npz') for i in range(6)]
            r['bilingual_route']='直接复用原PNG中的图像面板或已保存normalized_224_cam和相同输入，翻译标题/行列/图例；原make_*_gradcam入口会重新前后向，禁止为翻译执行。'
            g=V/'runs/MERGED_20261102/make_merged_trained_gradcam.py' if 'MERGED_20261102' in preferred.parts else V/('make_B_trained_gradcam.py' if 'B_20261102_TPE' in preferred.parts else 'make_A_trained_gradcam.py')
        elif rid=='D2-I15':
            r['evidence_sources']=[binding(parent/'spotcheck.json'),binding(parent/'spotcheck_independent_review.json')]
            g=V/'runs/MERGED_20261102/spotcheck_hand_detection.py'
        else:
            run=parent.parent; is_b='B_20261102_TPE' in preferred.parts
            if 'confusion' in preferred.name:
                r['evidence_sources']=[binding(run/('test_oracle_crop.json' if 'crop' in preferred.name else 'test_whole_photo.json'))]
                g=V/('postprocess_AB_results.py' if is_b else 'postprocess_A_results.py')
            elif 'learning_curves' in preferred.name:
                r['evidence_sources']=[binding(run/'events.jsonl')]
                r['required_evidence_note']='曲线须读取同协议各架构events.jsonl的validation事件，保留原epoch和F1；AB图同时读取A/B三架构。'
                g=V/('postprocess_AB_results.py' if is_b else 'postprocess_A_results.py')
            else:
                r['evidence_sources']=[binding(run/('actual_crop_comparison.json' if 'actual_crop' in preferred.name else 'error_examples.json'))]
                g=V/('make_B_error_examples.py' if is_b else 'make_A_error_examples.py')
        r['recommended_generators']=[binding(g)]
        r['generator_note']='已核为原生成脚本；应只读参考并另建只排版脚本，原入口可能写回科学文件。'

d['visual_review']={'method':'Inspect all 44 extracted image occurrences in 8 contact sheets; language/structural classification only, not full Word page rendering.','counts_unique':{'total':39,'photos_without_text':9,'external_english_originals':2,'project_chinese_technical_images':28},'current_completed_bilingual_pairs':0}
d['write_ownership_recommendation']=[
 {'owner':'technical_diagram_author','exclusive_output':'new zh/en native Visio copies, PNG exports and drawing scripts for 13 unique plan diagrams','dependency':'inventory source IDs and exact original PNGs'},
 {'owner':'result_plot_author','exclusive_output':'new zh/en plots and presentation-only Python for 15 unique experimental figures','dependency':'existing arrays/predictions/plot source evidence only; no model computation'},
 {'owner':'document_integrator','exclusive_output':'new plan v13 and report v4, optional new handover plus final ZIP','dependency':'both figure authors have complete two-language outputs and source map'},
 {'owner':'independent_reviewer','exclusive_output':'review record only','dependency':'actual image files, native text, source bindings, final Word render and ZIP contents'}]
d['scan_boundary_note']='最终候选搜索排除HaGRID、data、crops、train、val、test、images、dataset、全部渲染/预览/排版目录。初版因crops不叫data曾进入该目录并读取候选图进行SHA比较，随即停止并扩大剪枝；该意外目录信息未用于本盘点结论。'
(O/'figure_inventory.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 当前正式文档双语图与图源盘点','', '本轮只读核对三份正式Word，共44处可见图片，39个不同SHA256。当前正文尚未形成独立的中文版与英文版图对。9张无文字照片、2张外部英文原图、28张项目自制中文技术图。图内文字翻译和两版正文嵌入均须完成，翻译图注不能计为完成。','', '方案v12为29图，实验报告v3为15图，完整交接说明为0图。所有44处图片均找到项目内字节完全一致的源图片；原始嵌入图片已按SHA提取至embedded_originals。','', '13张方案自制图有现存可编辑VSDX，读取原生文字可追溯，但本轮未重新导出，导出像素身份待后续验证。15张不同实验图可用现有保存结果重新排版；4套正式Grad-CAM均有6份张量与metadata，不需要再运行模型。','', '两张电缆来源参考图（方案图10、图12）必须保留原英文图及来源，中文版只能标明是依据原图翻译的适配图；不能伪造外部作者的可编辑原图。无文字照片的中英图可复用同一照片像素，两语言版本分别保留在正文。','', '## 逐图状态','', '|图ID|现行图题|图内语言|同源制作路径|首选源|','|---|---|---|---|---|']
for r in d['figures']:
    src=Path(r['preferred_exact_source']['path']).relative_to(P).as_posix()
    route='原照片双语配套' if r['sequence']<=9 and r['id'].startswith('D1-') else '外部原图+忠实翻译适配' if r.get('external_original') else '原生VSDX复制翻译' if r['native_sources'] else '已保存证据只排版'
    lines.append('|'+ '|'.join([r['id'],r['caption'].replace('|','/'),r['image_text_language'],route,src])+'|')
lines += ['', '## 去重与分工依赖','', '正文有5张重复使用图：A正式Grad-CAM、B正式Grad-CAM、B裁图对照、合并A Grad-CAM、合并B Grad-CAM。每一对语言图只生成一次，供两份正式Word复用。','', '可并行进行方案技术图翻译和实验结果图排版；各自产出独占目录。两组完成后由单一整合者写新版Word并打ZIP，独立审者核图内所有文字、数值/科学关系、源文件与嵌入对应和实际Word页面。历史Word、冻结科学文件与当前上传旧ZIP均不修改。','', '详细逐图关系、SHA、源路径、原生文本、生成脚本、数值证据、重复组及限制见figure_inventory.json。','', '## 本轮检查边界','', d['scan_boundary_note'], '', '本轮未训练、未推理、未改原Word或原生图，未碰远端GitHub，也未操作旧ZIP上传。只检查图片内容与包结构，不声称已做Word完整页面渲染。']
(O/'figure_inventory.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Annotated',len(d['figures']),'occurrences; native',sum(bool(r['native_sources']) for r in d['figures']))
print('Missing evidence:',[(r['id'],Path(x['path']).name) for r in d['figures'] for x in r['evidence_sources'] if not x['exists']])
