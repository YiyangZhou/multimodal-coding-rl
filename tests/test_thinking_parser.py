import ast
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
from mmrl.config import validate_thinking_parser as validate

def recipe():
    return {'policy': {'generation': {'vllm_cfg': {'http_server_serving_chat_kwargs': {'reasoning_parser': 'deepseek_r1'}}}},
            'env': {'nemo_gym': {'policy_model': {'responses_api_models': {'vllm_model': {
                'extra_body': {'chat_template_kwargs': {'enable_thinking': True}}, 'uses_reasoning_parser': True}}}}}}

def model(cfg):
    return cfg['env']['nemo_gym']['policy_model']['responses_api_models']['vllm_model']

class Tests(unittest.TestCase):
    def test_parser_names_and_no_mutation(self):
        for name in ['deepseek_r1','qwen3','custom_registered_parser']:
            cfg=recipe();cfg['policy']['generation']['vllm_cfg']['http_server_serving_chat_kwargs']['reasoning_parser']=name
            before=copy.deepcopy(cfg);validate(cfg);self.assertEqual(cfg,before)
    def test_missing_or_invalid_parser(self):
        for value in [None,False,0,[],{},'', ' ', '${parser}']:
            cfg=recipe();cfg['policy']['generation']['vllm_cfg']['http_server_serving_chat_kwargs']['reasoning_parser']=value
            with self.subTest(value_type=type(value).__name__),self.assertRaisesRegex(ValueError,'resolved nonempty'):validate(cfg)
    def test_bad_nested_schema_is_safe_valueerror(self):
        paths=[['env'],['env','nemo_gym'],['env','nemo_gym','policy_model'],['env','nemo_gym','policy_model','responses_api_models'],['env','nemo_gym','policy_model','responses_api_models','vllm_model'],['policy'],['policy','generation'],['policy','generation','vllm_cfg'],['policy','generation','vllm_cfg','http_server_serving_chat_kwargs']]
        for keys in paths:
            for value in [None,[],True,'private-fixture-key']:
                cfg=recipe();parent=cfg
                for key in keys[:-1]:parent=parent[key]
                parent[keys[-1]]=value
                with self.assertRaises(ValueError) as caught:validate(cfg)
                self.assertNotIn('private-fixture-key',str(caught.exception))
            cfg=recipe();parent=cfg
            for key in keys[:-1]:parent=parent[key]
            del parent[keys[-1]]
            with self.assertRaisesRegex(ValueError,'Missing'):validate(cfg)
    def test_root_and_optional_mapping_types(self):
        for cfg in [None,[],True,'private-fixture-key']:
            with self.assertRaises(ValueError):validate(cfg)
        for field in ['extra_body','chat_template_kwargs']:
            for bad in [None,[],False,3,'private-fixture-key']:
                cfg=recipe();target=model(cfg) if field=='extra_body' else model(cfg)['extra_body'];target[field]=bad
                with self.assertRaisesRegex(ValueError,'must be a mapping'):validate(cfg)
    def test_non_thinking_and_legacy(self):
        for mode in ['false','missing_flag','missing_template','missing_extra']:
            cfg=recipe();cfg.pop('policy');m=model(cfg)
            if mode=='false':m['extra_body']['chat_template_kwargs']['enable_thinking']=False
            elif mode=='missing_flag':m['extra_body']['chat_template_kwargs'].pop('enable_thinking')
            elif mode=='missing_template':m['extra_body'].pop('chat_template_kwargs')
            else:m.pop('extra_body')
            before=copy.deepcopy(cfg);validate(cfg);self.assertEqual(cfg,before)
    def test_bool_contracts(self):
        for value in [None,0,1,'true',[],{}]:
            cfg=recipe();model(cfg)['extra_body']['chat_template_kwargs']['enable_thinking']=value
            with self.assertRaisesRegex(ValueError,'boolean'):validate(cfg)
        for value in [None,False,0,1,'true']:
            cfg=recipe();model(cfg)['uses_reasoning_parser']=value
            with self.assertRaisesRegex(ValueError,'uses_reasoning_parser=true'):validate(cfg)
        cfg=recipe();model(cfg).pop('uses_reasoning_parser');validate(cfg)
    def test_cli_fails_before_credentials_or_capture_validation(self):
        calls=[]
        config=types.ModuleType('mmrl.config');config.validate_thinking_parser=validate
        config.validate_capture_retries=lambda cfg:calls.append('capture')
        config.validate_sibling_recipe=lambda cfg:calls.append('sibling')
        sandbox=types.ModuleType('mmrl.sandbox');sandbox.provider_config=lambda:calls.append('credentials')
        cfg=recipe();cfg['policy']['generation']['vllm_cfg']['http_server_serving_chat_kwargs']={}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'cfg.json';path.write_text(json.dumps(cfg))
            with patch.dict(sys.modules,{'mmrl.config':config,'mmrl.sandbox':sandbox}):
                module=runpy.run_path(str(HERE/'scripts/train.py'))
                with patch.object(sys,'argv',['train.py','--config',str(path),'--rl-root',td,'--check-only']):
                    with self.assertRaisesRegex(ValueError,'reasoning_parser'):module['main']()
        self.assertEqual(calls,[])

if __name__=='__main__':unittest.main(verbosity=2)
