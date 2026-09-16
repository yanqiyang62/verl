from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
P=Path(__file__).resolve().parent
s=json.loads((P/'assets/summary.json').read_text()); rows=list(csv.DictReader((P/'assets/step-metrics.csv').open()))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
xs=[int(r['step']) for r in rows]; ys=[float(r['critic/score/mean']) for r in rows]
axs[0,0].plot(xs,ys,alpha=.32,lw=.8,color='#2463a5',label='Per-step reward')
axs[0,0].plot([w['end'] for w in s['windows']],[w['mean_reward'] for w in s['windows']],color='#d66a24',lw=2,label='Window mean')
axs[0,0].set(title='A. Training reward (different prompts)',xlabel='Step',ylabel='Reward',ylim=(-.03,1.05));axs[0,0].legend()
win=s['windows'];axs[0,1].bar([f"{w['start']}-{w['end']}" for w in win],[100*w['nonzero_adv_steps']/(w['end']-w['start']+1) for w in win],color='#278576')
axs[0,1].set(title='B. Steps with nonzero advantages',ylabel='Share (%)',ylim=(0,100));axs[0,1].tick_params(axis='x',rotation=55)
cats=['parse_or_type_error','call_or_schema_rejected','parsed_zero','partial','exact']; labels=['Parse/type error','Call/schema rejected','Parsed, zero','Partial reward','Exact reference'];colors=['#ba4945','#da8260','#ddb85d','#58a499','#2865a2'];bottom=[0,0]
for cat,label,color in zip(cats,labels,colors):
 vals=[s['by_kind'][k]['categories'].get(cat,0) for k in ['json','tool_call']];axs[1,0].bar(['JSON (472)','Tool call (486)'],vals,bottom=bottom,label=label,color=color);bottom=[a+b for a,b in zip(bottom,vals)]
axs[1,0].set(title='C. Readable answers: reward outcomes',ylabel='Answers');axs[1,0].legend(fontsize=8)
t=s['timing_sums'];total=t['timing_s/step'];names=['Generate','Checkpoint','Weight sync','Actor update','Other'];values=[t[k] for k in ['timing_s/gen','timing_s/save_checkpoint','timing_s/update_weights','timing_s/update_actor']];values.append(total-sum(values))
axs[1,1].barh(names[::-1],[v/3600 for v in values[::-1]],color='#607e9d');axs[1,1].set(title='D. Logged step-time breakdown',xlabel='Hours')
for i,v in enumerate(values[::-1]):axs[1,1].text(v/3600+.025,i,f'{100*v/total:.1f}%',va='center')
fig.suptitle('Qwen3.8 native GRPO | run psmh3kg8 | steps 1-480',fontsize=15)
fig.savefig(P/'assets/overview.png',dpi=160);fig.savefig(P/'assets/overview.svg');plt.close(fig)
body=markdown.markdown((P/'README.md').read_text(),extensions=['tables','fenced_code','toc'])
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen3.8 GRPO 本轮训练分析</title><style>body{font-family:system-ui,"Noto Sans SC",sans-serif;color:#243346;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.8;background:#fafbfd}h1,h2,h3{line-height:1.4;color:#153452}h2{margin-top:2.4em;border-bottom:2px solid #dce6ef;padding-bottom:10px}a{color:#1464aa}table{border-collapse:collapse;width:100%;display:block;overflow:auto;margin:20px 0;font-size:.94em}th,td{padding:10px 14px;border:1px solid #dce4ec;text-align:left}th{background:#e9f0f7}tr:nth-child(even){background:#f2f6fa}img{max-width:100%;height:auto}pre{overflow:auto;background:#eaf0f6;padding:18px}code{overflow-wrap:anywhere;font-size:.9em}blockquote{border-left:4px solid #348d81;margin:20px 0;padding:8px 18px;background:#edf5f2}@media print{body{max-width:none;margin:0}h2{break-after:avoid}table{display:table;font-size:9pt}a{color:inherit}}</style><main>'''+body+'</main></html>'
(P/'report.html').write_text(html)
print('Generated overview.png, overview.svg, report.html')
