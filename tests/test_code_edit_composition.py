"""Exercise all three render paths and fail-closed source handling."""
import asyncio,json
from pathlib import Path
from types import SimpleNamespace
from mmrl.service import CodingService
from mmrl.tasks import Task
from mmrl.reward import InfrastructureError
from mmrl.web_document import compose_document
class Sandbox:
 def __init__(self):self.calls=[]
 async def render(self,benchmark,source):self.calls.append(source);return b'png'
class Judge:
 def __init__(self):self.called=False
 async def grade(self,*args,**kwargs):
  self.called=True
  assert 'original_render' in kwargs and 'target_render' in kwargs
  return SimpleNamespace(reason='',reward=.8,task_fidelity=.75,visual_fidelity=.75)
async def main():
 source='```html\n<html><head></head><body><p>Text</p></body></html>\n``````css\np { color: blue; }\n```'
 patch='-p { color: blue; }\n+p { color: black; }'
 task=Task('webmmu_code_edit','test','Change color',original_source=source,reference_patch=patch)
 box=Sandbox();judge=Judge();result=await CodingService(box,judge).score(task,patch)
 assert result['execution']==1 and len(box.calls)==3 and judge.called
 assert all('```' not in document and '<style>' in document for document in box.calls)
 assert 'color: black' in box.calls[0] and 'color: blue' in box.calls[1] and 'color: black' in box.calls[2]
 missing=Task('webmmu_code_edit','missing','Change color',original_source='```css\np { color: blue; }\n```',reference_patch=patch)
 box=Sandbox();judge=Judge()
 try:await CodingService(box,judge).score(missing,patch)
 except InfrastructureError:pass
 else:raise AssertionError('Missing source must not be scored zero')
 assert not box.calls and not judge.called
 javascript='```html\n<html><body><p id="x">A</p></body></html>\n``````javascript\ndocument.getElementById("x").textContent="B";\n```'
 assert '<script>' in compose_document(javascript)
 raw='<html><body>unchanged</body></html>';assert compose_document(raw)==raw

def test_service_contract():
 asyncio.run(main())
