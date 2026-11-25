"""Join committed train rows to verifier records; missing evidence is never zero."""
import math
import re
from collections import defaultdict

def scalar(row,key):
 value=row[key]
 if not isinstance(value,list) or len(value)!=1:raise ValueError(f'Invalid {key}')
 value=float(value[0])
 if not math.isfinite(value):raise ValueError(f'Nonfinite {key}')
 return value

def summarize(rows,records):
 index=defaultdict(list)
 physical=defaultdict(list)
 for record in records:
  if not record.get('complete'):continue
  rid=record.get('rollout_id','')
  physical[rid].append(record)
  match=re.fullmatch(r'(.+_g[0-9]+)_([0-9a-f]{32})',rid)
  if match:index[match[1]].append(record)
 groups=defaultdict(list);execution=[];raw=[];infra=0;valid=0;seen=set();steps=set();missing=[]
 for row in rows:
  sid=row['sample_id'][0];steps.add(row['step'])
  if sid in seen:raise ValueError('Duplicate consumed sample')
  seen.add(sid)
  if scalar(row,'sample_loss_mask')>0:valid+=1
  tags=row.get('metadata',[{}])[0]
  rid=tags.get('verification_rollout_id')
  candidates=physical.get(rid,[]) if rid else index.get(sid,[])
  if len(candidates)!=1:
   missing.append({'sample_id':sid,'reason':'missing' if not candidates else 'ambiguous'})
   continue
  record=candidates[0];result=record['result']
  if result.get('mask_sample') or result.get('failure_kind')=='mm_coding:infrastructure':
   infra+=1;continue
  reward=scalar(row,'raw_quality_reward')
  if not math.isclose(reward,float(result['reward']),rel_tol=1e-5,abs_tol=1e-6):raise ValueError('Verifier and consumed raw reward disagree')
  value=result.get('execution')
  if value not in (0,1):raise ValueError('Execution outcome unavailable')
  raw.append(reward);execution.append(value);groups[record['benchmark']].append(reward)
 if len(steps)!=1:raise ValueError('Expected one committed optimizer step')
 metrics={'overview/valid_samples':valid,'overview/infrastructure_failures':infra,'overview/evidence_missing':len(missing),'overview/observed_samples':len(seen)}
 # Do not present incomplete evidence as full-batch aggregates.
 if not missing:
  if raw:metrics['overview/raw_reward']=sum(raw)/len(raw)
  if execution:metrics['overview/execution_success_rate']=sum(execution)/len(execution)
  for name,values in groups.items():metrics[f'overview/task_reward/{name}']=sum(values)/len(values)
 return {'optimizer_step':next(iter(steps)),'complete':not missing,'metrics':metrics,'missing':missing,'task_counts':{k:len(v) for k,v in groups.items()}}


def metrics_for_step(log_directory, verification_directory, step):
    import json
    from pathlib import Path
    path = Path(log_directory) / f"train_data_step{step}.jsonl"
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    records = []
    seen = set()
    for row in rows:
        rid = row.get("metadata", [{}])[0].get("verification_rollout_id")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", rid):
            continue
        if rid in seen:
            continue
        seen.add(rid)
        for record in (Path(verification_directory) / "by-rollout" / rid).glob("*.json"):
            records.append(json.loads(record.read_text()))
    return summarize(rows, records)
