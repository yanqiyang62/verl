from pathlib import Path
import collections,csv,datetime,json,re,statistics,sys
OUT=Path(__file__).resolve().parent/'assets';ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(OUT/'source'))
from tf_rl import reward
examples=json.loads((OUT/'examples-all.json').read_text()); cf=[]
for r in examples:
 raw=json.loads((OUT/'source/rollouts'/f"{r['step']}.jsonl").read_text().splitlines()[r['sample']-1]);spec=reward.ground_truth_spec(raw['gts']);s=r['score'];f=reward.final_text(r['output']); mode='unchanged'
 if r['kind']=='json':
  match=re.fullmatch(r'<response>\s*(.*?)\s*</response>',f,re.S)
  if match:
   mode='response_wrapper';s=reward.compute_score('',match[1],raw['gts'])['score']
  else:
   match=re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```',f)
   if match:mode='whole_fence';s=reward.compute_score('',match[1],raw['gts'])['score']
 cf.append({'step':r['step'],'sample':r['sample'],'mode':mode,'before':r['score'],'after':s})
groups=collections.defaultdict(list)
for r in cf:groups[r['step']].append(r['after'])
summary={'note':'Counterfactual packaging-only diagnostic, not a deployed reward change or semantic validation','n':len(cf),'before_mean':statistics.mean(r['before'] for r in cf),'after_mean':statistics.mean(r['after'] for r in cf),'positive_after':sum(r['after']>0 for r in cf),'different_groups_after':sum(max(g)-min(g)>1e-7 for g in groups.values()),'by_mode':{m:{'n':len(rs),'positive_after':sum(r['after']>0 for r in rs),'exact_after':sum(r['after']==1 for r in rs),'changed':sum(abs(r['after']-r['before'])>1e-7 for r in rs)} for m in ['response_wrapper','whole_fence'] if (rs:=[r for r in cf if r['mode']==m])}}
(OUT/'packaging-counterfactual.json').write_text(json.dumps(summary,indent=2))
with (OUT/'packaging-counterfactual.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(cf[0]));w.writeheader();w.writerows(cf)
# Bounded, read-only integrity inspection; never deserialize checkpoints.
c=ROOT/'checkpoints/GRPO-Qwen3.8-Smoke/Qwen3.8-27B-LoRA-FSDP2-MIG67G-ReferenceReward';checks=[]
for f in [c/'latest_checkpointed_iteration.txt']+list((c/'global_step_460').rglob('*'))+list((c/'global_step_480').rglob('*')):
 if not f.is_file():continue
 n=f.stat().st_size
 with f.open('rb') as h:b=h.read(n if n<100000 else 64)
 checks.append({'path':str(f.relative_to(ROOT)),'bytes':n,'read_bytes':len(b),'all_inspected_bytes_zero':bool(b) and not b.strip(b'\0'),'full_file_inspected':len(b)==n})
(OUT/'checkpoint-integrity.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Full read below 100000 bytes, otherwise first 64 bytes only; no full model load or checksum','files':checks},indent=2))
selected=[]
for step,sample in [(12,2),(20,1),(11,2),(200,1),(44,1),(402,1),(419,1)]:
 r=next(r for r in examples if r['step']==step and r['sample']==sample);selected.append(r)
(OUT/'examples-selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
ms=list(csv.DictReader((OUT/'step-metrics.csv').open()))
print(json.dumps(summary,indent=2))
print('first221',statistics.mean(float(m['critic/score/mean']) for m in ms[:221]),sum(float(m['critic/advantages/max'])!=0 for m in ms[:221]))
for step in [402,419]:
 m=ms[step-1];print('anomaly',step,{k:m[k] for k in ['critic/score/mean','actor/entropy','rollout_corr/k3_kl','training/rollout_probs_diff_mean','response_length/mean']})
print('positive and equal',sum(float(m['critic/score/max'])>0 and float(m['critic/score/max'])==float(m['critic/score/min']) for m in ms))
print('file integrity',[(r['path'].split('/')[-1],r['all_inspected_bytes_zero']) for r in checks if r['full_file_inspected']])
