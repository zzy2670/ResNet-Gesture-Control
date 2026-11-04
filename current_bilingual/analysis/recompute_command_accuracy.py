"""仅用冻结逐图预测重算整图命令准确率差额，不读取照片或运行模型。"""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parent

def main():
    paths = [ROOT/'inputs/B18_unmerged_whole.json', ROOT/'inputs/B18_merged_whole.json']
    data = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
    old, new = [{r['image_id']: r for r in d['predictions']} for d in data]
    assert len(old)==len(new)==3102 and set(old)==set(new)
    changed = [k for k in old if old[k]['truth'] != new[k]['truth']]
    unchanged = [k for k in old if old[k]['truth'] == new[k]['truth']]
    assert len(changed)==336 and all(old[k]['truth']==0 and new[k]['truth']==1 for k in changed)
    def correct(rows, keys):
        return sum(rows[k]['truth']==rows[k]['prediction'] for k in keys)
    groups = [{'cohort':name,'n':len(keys),'old_correct':correct(old,keys),'new_correct':correct(new,keys)} for name,keys in [('newly_required_two_up',changed),('unchanged_truth',unchanged),('all',list(old))]]
    failed = [k for k in changed if new[k]['prediction'] != new[k]['truth']]
    no_boxes = [k for k in failed if not new[k]['hands']]
    assert len(failed)==29 and len(no_boxes)==19 and all(new[k]['prediction']==0 for k in failed)
    assert all(not any(h['accepted'] and h['label']!=0 for h in new[k]['hands']) for k in failed)
    negative = [k for k in unchanged if old[k]['truth']==new[k]['truth']==0]
    motion={1,2,4,5}
    false_motion=[sum(rows[k]['prediction'] in motion for k in negative) for rows in [old,new]]
    result={'status':'RECOMPUTED_FROM_FROZEN_PREDICTIONS','groups':groups,'newly_required_failed':29,'no_detected_boxes':19,'boxes_without_accepted_nonzero_command':10,'failed_image_ids':failed,'no_boxes_image_ids':no_boxes,'unchanged_old_correct_new_wrong':sum(old[k]['prediction']==old[k]['truth'] and new[k]['prediction']!=new[k]['truth'] for k in unchanged),'unchanged_old_wrong_new_correct':sum(old[k]['prediction']!=old[k]['truth'] and new[k]['prediction']==new[k]['truth'] for k in unchanged),'same_negative_photos':len(negative),'false_motion_old_new':false_motion,'thresholds_old_new':[{'detector':d['detector_threshold'],'classifier':d['class_threshold']} for d in data],'sources':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],'training_inference_or_image_access':False}
    assert [(g['old_correct'],g['new_correct']) for g in groups]==[(336,307),(2551,2542),(2887,2849)]
    assert len(negative)==1359 and false_motion==[5,16]
    output=ROOT/'whole_command_accuracy_B18.json'
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
