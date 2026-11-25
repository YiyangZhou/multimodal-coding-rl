"""Ensure renderer repair does not turn verifier outages into model failures."""
import asyncio,json
from pathlib import Path
from types import SimpleNamespace
from mmrl.service import CodingService
from mmrl.tasks import Task
from mmrl.reward import InfrastructureError

class Box:
 def __init__(self,results):self.results=iter(results);self.calls=0
 async def render(self,*args):
  self.calls+=1;r=next(self.results)
  if isinstance(r,Exception):raise r
  return r
class Judge:
 def __init__(self,fail=False):self.calls=0;self.fail=fail
 async def grade(self,*args,**kwargs):
  self.calls+=1
  if self.fail:raise InfrastructureError('simulated API outage')
  return SimpleNamespace(reason='',reward=.8,task_fidelity=.75,visual_fidelity=.75)

async def main():
 src='```html\n<html><head></head><body><p>A</p></body></html>\n``````css\np {color:red;}\n```'
 patch='-p {color:red;}\n+p {color:blue;}'
 task=Task('webmmu_code_edit','boundary','Change color',original_source=src,reference_patch=patch)
 rows=[]
 async def check(name,answer,results,expected,judge_fail=False):
  box=Box(results);judge=Judge(judge_fail)
  try:
   r=await CodingService(box,judge).score(task,answer)
   assert expected=='model_failure' and r['reward']==0 and r['execution']==0
   assert judge.calls==0
  except InfrastructureError:
   assert expected=='infrastructure'
  rows.append({'case':name,'classification':expected,'render_calls':box.calls,'judge_calls':judge.calls,'passed':True})
 await check('nonmatching_diff','-not in source\n+replacement',[],'model_failure')
 await check('candidate_render_failure',patch,[None],'model_failure')
 await check('candidate_transport_failure',patch,[InfrastructureError('transport')],'infrastructure')
 await check('original_render_failure',patch,[b'candidate',None],'infrastructure')
 await check('target_transport_failure',patch,[b'candidate',b'original',InfrastructureError('transport')],'infrastructure')
 await check('judge_api_failure',patch,[b'candidate',b'original',b'target'],'infrastructure',True)

def test_service_contract():
 asyncio.run(main())
