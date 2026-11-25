"""CPU checks requiring the configured native NeMo Gym dependency environment."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from mmrl.gym_app import CodingResourcesServer, VerifyRequest
from mmrl.tasks import Task
from responses_api_models.vllm_model.app import VLLMModelConfig

class NativeTests(unittest.IsolatedAsyncioTestCase):
    def test_native_model_schema(self):
        config = VLLMModelConfig(host='localhost', port=1, entrypoint='app.py', name='fixture', base_url='http://localhost:1', api_key='fixture', model='fixture', return_token_id_information=True, uses_reasoning_parser=True, extra_body={'chat_template_kwargs':{'enable_thinking':True}})
        self.assertIs(config.uses_reasoning_parser, True)
        self.assertIs(config.extra_body['chat_template_kwargs']['enable_thinking'], True)
        # Native class requires this field; the training YAML supplies its default.
        self.assertTrue(VLLMModelConfig.model_fields['uses_reasoning_parser'].is_required())

    async def test_real_verify_preserves_incomplete_and_failure_masks(self):
        task=Task('mmsvg_text2svg','fixture','fixture prompt')
        for status,metadata,error,expected in [
            ('incomplete',{'partial':'true','failed':'false'},None,'mm_coding:incomplete'),
            ('failed',{'partial':'true','failed':'true'},{'code':'server_error','message':'PRIVATE_ERROR'},'mm_coding:infrastructure'),
        ]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                body=VerifyRequest.model_validate({'responses_create_params':{'input':'fixture'},'capture_rollout_id':'rollout-test','task':asdict(task),'route':'hermes','response':{'id':'response-test','created_at':0,'model':'fixture','object':'response','status':status,'metadata':{**metadata,'hermes_error':'PRIVATE_ERROR'},'error':error,'output':[],'tool_choice':'auto','tools':[],'parallel_tool_calls':True}})
                session=SimpleNamespace(task=task,route='hermes',capture_rollout_id='rollout-test',handle=None)
                server=SimpleNamespace(_pop_session=lambda _:session, _before_verify=lambda _:None, config=SimpleNamespace(verification_log_dir=directory,media_capture_task_ids=[]))
                result=await CodingResourcesServer.verify(server,SimpleNamespace(session={'mm_session':'fixture'}),body)
                self.assertEqual(result.reward,0)
                self.assertIs(result.mask_sample,True)
                self.assertIs(result.instance_config['mask_sample'],True)
                self.assertEqual(result.failure_kind,expected)
                saved=json.loads(next(Path(directory).glob('*/record.json')).read_text())
                self.assertEqual(saved['upstream_terminal'],{'partial':True,'failed':status=='failed'})
                self.assertNotIn('PRIVATE_ERROR',json.dumps(saved))

if __name__=='__main__': unittest.main(verbosity=2)
