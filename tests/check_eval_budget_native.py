import asyncio,copy,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import mmrl.evaluation_runtime as n
from fastapi import FastAPI
import httpx
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from responses_api_models.vllm_model.app import VLLMModel
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming

original=n.evaluation_classes
class Tests(unittest.IsolatedAsyncioTestCase):
 async def asyncSetUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.l=n.BudgetLedger(self.tmp.name,10,20,100)
  self.r='a'*32;self.l.register(self.r,'case-1');self.row=self.l.rows[self.r]
  n.evaluation_classes=original
  # Mimic the existing assets factory BEFORE installing the overlay.
  def assets(ledger):
   model,resource=original(ledger)
   class AssetsResources(resource):
    async def seed_session(self,request,body):return 'assets-preserved'
   return model,AssetsResources
  n.evaluation_classes=assets
  self.Model,self.Resource=n.evaluation_classes(self.l)
  with patch.object(self.Model,'model_post_init',lambda *a:None):
   self.model=self.Model.model_construct(config=types.SimpleNamespace(name='fixture'))
 async def asyncTearDown(self):self.tmp.cleanup();n.evaluation_classes=original
 async def invoke(self, *, count=10, usage=3, upstream_error=False, stream=False):
  async def tokenize(**kwargs):return {'count':count}
  async def upstream(*args):
   if upstream_error:raise RuntimeError('upstream fixture error')
   return types.SimpleNamespace(model_dump=lambda **kw:{'usage':{'completion_tokens':usage},'choices':[{'finish_reason':'length'}]})
  body=NeMoGymChatCompletionCreateParamsNonStreaming(model='fixture',messages=[{'role':'user','content':'fixture'}])
  with patch.object(VLLMModel,'_preprocess_chat_completion_create_params',lambda s,r,b:b),patch.object(VLLMModel,'_get_tokenize_chat_body',lambda s,b:b),patch.object(VLLMModel,'_resolve_client',lambda s,r:types.SimpleNamespace(create_tokenize=tokenize)),patch.object(VLLMModel,'chat_completions',upstream):
   # Actual Gym dispatch/converter and correlation middleware, ASGI transport.
   app=FastAPI();app.add_middleware(RolloutContextMiddleware)
   app.post('/v1/chat/completions')(self.model.chat_completions_dispatch)
   async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,raise_app_exceptions=False),base_url='http://fixture') as client:
    raw=body.model_dump(exclude_none=True);raw['stream']=stream
    return await client.post('/ng-rollout/'+self.r+'/v1/chat/completions',json=raw)
 async def test_output_terminal_repeated_stream_and_nonstream(self):
  self.row.update(output_tokens=20,calls=2)
  for stream in (False,True,False):
   response=await self.invoke(stream=stream)
   self.assertEqual(response.status_code,400,response.text)
  self.assertTrue(self.row['truncated']);self.assertFalse(self.row['infrastructure_error']);self.assertEqual(self.row['calls'],2)
 async def test_context_terminal(self):
  response=await self.invoke(count=100)
  self.assertEqual(response.status_code,400,response.text);self.assertEqual(self.row['budget_terminal_reason'],'context_window');self.assertFalse(self.row['infrastructure_error'])
 async def test_upstream_error_not_hidden_by_truncation(self):
  self.row['truncated']=True
  response=await self.invoke(upstream_error=True)
  self.assertEqual(response.status_code,500);self.assertTrue(self.row['infrastructure_error'])
 async def test_invalid_usage_remains_infra(self):
  self.row['truncated']=True
  response=await self.invoke(usage=999)
  self.assertEqual(response.status_code,500);self.assertTrue(self.row['infrastructure_error']);self.assertNotIn('budget_terminal_reason',self.row)
 async def test_prior_infra_cannot_become_budget_terminal(self):
  self.row.update(output_tokens=20,infrastructure_error=True)
  response=await self.invoke();self.assertEqual(response.status_code,500);self.assertNotIn('budget_terminal_reason',self.row)
 async def test_assets_wrapper_preserved(self):
  with patch.object(self.Resource,'model_post_init',lambda *a:None):
   resource=self.Resource.model_construct()
  self.assertEqual(await resource.seed_session(None,None),'assets-preserved')

 async def test_verify_precedence_and_transport_reward(self):
  from mmrl.tasks import Task
  from dataclasses import asdict
  task=Task('mmsvg_text2svg','fixture','fixture prompt')
  session=types.SimpleNamespace(capture_rollout_id=self.r,task=task,route='fixture',handle=None)
  async def artifact(*a):return '<svg/>'
  fake=types.SimpleNamespace(_pop_session=lambda _:session,_sandbox=types.SimpleNamespace(read_artifact=artifact))
  response=types.SimpleNamespace(status='failed',error={'type':'terminal fixture'},metadata={},model_dump=lambda **kw:{'status':'failed'})
  body=types.SimpleNamespace(capture_rollout_id=self.r,route='fixture',task=asdict(task),response=response,model_dump=lambda :{})
  request=types.SimpleNamespace(session={'mm_session':'fixture'})
  method=self.Resource.verify
  # Keep the exact native verify body; replace only output transport constructor.
  import inspect
  names=method.__code__.co_freevars
  cell=method.__closure__[names.index('VerifyResponse')]
  old=cell.cell_contents;cell.cell_contents=lambda **kw:types.SimpleNamespace(**kw)
  try:
   for terminal,infra,expected in [(True,False,'truncated'),(True,True,'infrastructure_error'),(False,False,'infrastructure_error')]:
    self.row.update(truncated=True,infrastructure_error=infra,calls=2,output_tokens=20)
    self.row.pop('budget_terminal_reason',None)
    if terminal:self.row['budget_terminal_reason']='output_budget'
    result=await method(fake,request,body)
    self.assertEqual(result.evaluation_record['status'],expected)
    self.assertEqual(result.reward,0.0);self.assertIs(result.mask_sample,True)
  finally:cell.cell_contents=old

if __name__=='__main__':unittest.main(verbosity=2)
