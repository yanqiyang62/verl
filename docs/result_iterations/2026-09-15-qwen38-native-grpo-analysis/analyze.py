"""Freeze local evidence and recompute reward diagnostics; no training mutations."""
from pathlib import Path
import collections,csv,datetime,hashlib,json,re,shutil,sys,statistics
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'assets'
SRC=OUT/'source'
RECIPE=Path('/shared/users/yangyq/data/recipe_v2_qwen38_native')
LOG=ROOT/'logs/Qwen3.8-27B-LoRA-FSDP2-MIG67G-ReferenceReward-20260915_020721.log'
# Reuse the first snapshot on reruns.
if not (SRC/'training.log').exists():
 shutil.copyfile(LOG,SRC/'training.log')
 for name in ['reward.py','qwen_tool_parser.py','agent_loop.py','dataset.py','rendering.py']:
  (SRC/'tf_rl').mkdir(exist_ok=True); shutil.copyfile(RECIPE/'tf_rl'/name,SRC/'tf_rl'/name)
 for name in ['config.yaml','overrides.yaml']:
  shutil.copyfile(ROOT/'outputs/2026-09-15/02-07-38/.hydra'/name,SRC/name)
 (SRC/'captured_at.txt').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())
sys.path.insert(0,str(SRC))
from tf_rl import reward
text=(SRC/'training.log').read_text()
metrics=[]
for line in text.splitlines():
 m=re.search(r'\bstep:(\d+) - ',line)
 if not m:continue
 row={'step':int(m[1])}
 for k,v in re.findall(r'([\w/.-]+):(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)',line[m.end():]):row[k]=float(v)
 metrics.append(row)
assert [m['step'] for m in metrics]==list(range(1,len(metrics)+1))
rollout=ROOT/'logs/rollouts/Qwen3.8-27B-LoRA-FSDP2-MIG67G-ReferenceReward'
manifest=[]; answers=[]; examples=[]; groups=[]
for m in metrics:
 step=m['step']; p=rollout/f'{step}.jsonl'; dst=SRC/'rollouts'/p.name; dst.parent.mkdir(exist_ok=True)
 if not dst.exists():shutil.copyfile(p,dst)
 b=dst.read_bytes(); meta={'step':step,'path':str(p.relative_to(ROOT)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'nul_bytes':b.count(b'\0')}
 try:
  rows=[json.loads(line) for line in b.splitlines() if line.strip()]
  assert len(rows)==2 and all(int(r['step'])==step for r in rows)
 except (ValueError,AssertionError) as e:
  meta['status']='unreadable'; manifest.append(meta);continue
 meta['status']='valid';manifest.append(meta)
 scores=[float(r['score']) for r in rows]
 assert abs(statistics.mean(scores)-m['critic/score/mean'])<1e-5,(step,scores)
 groups.append({'step':step,'score_1':scores[0],'score_2':scores[1],'different':abs(scores[0]-scores[1])>1e-8,'identical_output':rows[0]['output']==rows[1]['output']})
 for i,r in enumerate(rows,1):
  spec=reward.ground_truth_spec(r['gts']); calc=reward.compute_score('',r['output'],r['gts']); err=''; pred=None
  assert abs(calc['score']-r['score'])<1e-5,(step,i,calc,r['score'])
  try:pred=reward.parse_output(r['output'],spec['kind'],spec.get('allow_fence',False),spec.get('schemas'))
  except Exception as e:err=type(e).__name__+': '+str(e).splitlines()[0]
  if err:category='parse_or_type_error'
  elif not calc['format_valid']:category='call_or_schema_rejected'
  elif r['score']==0:category='parsed_zero'
  elif calc['acc']:category='exact'
  else:category='partial'
  wrapped_exact=False
  f=reward.final_text(r['output'])
  wrap=re.fullmatch(r'<response>\s*(.*?)\s*</response>',f,re.S)
  if wrap and spec['kind']=='json':
   try:wrapped_exact=reward.canonical(reward.strict_json(wrap[1]))==reward.canonical(spec['expected'])
   except ValueError:pass
  row={'step':step,'sample':i,'uid':r['uid'],'kind':spec['kind'],'score':r['score'],'category':category,'error':err,'format_valid':calc['format_valid'],'field_accuracy':calc['field_accuracy'],'acc':calc['acc'],'response_wrapper':bool(wrap),'wrapper_inner_exact':wrapped_exact,'xml_output':'<function=' in r['output'],'expected_tools':','.join(c['name'] for c in spec['expected']) if spec['kind']=='tool_call' else '', 'expected_keys':','.join(spec['expected']) if isinstance(spec['expected'],dict) else ''}
  answers.append(row)
  examples.append(dict(row,output=r['output'],expected=spec['expected'],input=r['input'],parsed=pred))

def writejson(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
def writecsv(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def stats(rows):
 return {'n':len(rows),'mean_score':statistics.mean(r['score'] for r in rows) if rows else None,'positive':sum(r['score']>0 for r in rows),'categories':dict(collections.Counter(r['category'] for r in rows))}
windows=[]
for start in range(1,len(metrics)+1,50):
 ms=[m for m in metrics if start<=m['step']<start+50];ars=[r for r in answers if start<=r['step']<start+50]
 windows.append({'start':start,'end':ms[-1]['step'],'mean_reward':statistics.mean(m['critic/score/mean'] for m in ms),'nonzero_adv_steps':sum(m['critic/advantages/max']!=0 or m['critic/advantages/min']!=0 for m in ms),'entropy':statistics.mean(m['actor/entropy'] for m in ms),'readable_answers':len(ars),'tool_answers':sum(r['kind']=='tool_call' for r in ars)})
summary={'snapshot_at':(SRC/'captured_at.txt').read_text(),'steps':len(metrics),'mean_reward':statistics.mean(m['critic/score/mean'] for m in metrics),'positive_steps':sum(m['critic/score/max']>0 for m in metrics),'nonzero_adv_steps':[m['step'] for m in metrics if m['critic/advantages/max']!=0 or m['critic/advantages/min']!=0],'nonzero_pg_steps':[m['step'] for m in metrics if m['actor/pg_loss']!=0],'readable':stats(answers),'by_kind':{k:stats([r for r in answers if r['kind']==k]) for k in ['json','tool_call']},'unreadable_steps':[r['step'] for r in manifest if r['status']!='valid'],'different_groups':sum(r['different'] for r in groups),'identical_output_groups':sum(r['identical_output'] for r in groups),'parse_errors':collections.Counter(r['error'] for r in answers if r['error']),'wrapper_count':sum(r['response_wrapper'] for r in answers),'wrapper_exact':sum(r['wrapper_inner_exact'] for r in answers),'windows':windows,'timing_sums':{k:sum(m.get(k,0) for m in metrics) for k in ['timing_s/step','timing_s/gen','timing_s/save_checkpoint','timing_s/update_actor','timing_s/update_weights']},'max_response':max(m['response_length/max'] for m in metrics),'metric_ranges':{k:{'min':min(m[k] for m in metrics),'max':max(m[k] for m in metrics),'mean':statistics.mean(m[k] for m in metrics)} for k in ['actor/grad_norm','actor/kl_loss','actor/pg_clipfrac','training/rollout_actor_probs_pearson_corr','actor/perf/max_memory_allocated_gb','actor/perf/max_memory_reserved_gb']}}
writejson('summary.json',summary);writejson('rollout-manifest.json',manifest);writejson('examples-all.json',examples)
writecsv('step-metrics.csv',metrics);writecsv('answer-evidence.csv',answers);writecsv('group-evidence.csv',groups);writecsv('windows.csv',windows)
writejson('source-hashes.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
print(json.dumps(summary,ensure_ascii=False,indent=2))
