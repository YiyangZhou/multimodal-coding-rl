import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mmrl.cli import main
from mmrl.tasks import Task, write_rows

class MixedComposeTest(unittest.TestCase):
    def test_compose_preserves_weights_and_rejects_wrong_group_size(self):
        base={'data_plane':{'enabled':True},'async_rl':{},
              'policy':{'generation':{'temperature':.8,'top_p':.9,'top_k':None,
                         'max_new_tokens':32768,'vllm_cfg':{'max_model_len':65536}}},
              'data':{'max_input_seq_length':65536,'train':{'seed':123,
                      'benchmark_weights':{'mmsvg_text2svg':2},'balance_batch_size':8}},
              'grpo':{'num_generations_per_prompt':8},'checkpointing':{},'logger':{}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data.jsonl';cfg=root/'base.json'
            write_rows([Task('mmsvg_text2svg','one','Draw a circle.')],data,'sibling-mixed')
            cfg.write_text(json.dumps(base))
            argv=['mmrl','compose','--base',str(cfg),'--data',str(data),'--image','image@sha256:'+'a'*64,'--gateway','auto','--output',str(root/'run')]
            with patch('sys.argv',argv): main()
            train=json.loads((root/'run/train.json').read_text())['data']['train']
            self.assertEqual(train['dataset_name'],'mmrl.mixed_dataset.BalancedMixedGymDataset')
            self.assertEqual(train['benchmarks'],['mmsvg_text2svg'])
            self.assertEqual(train['benchmark_weights'],{'mmsvg_text2svg':2})
            self.assertEqual(train['seed'],123)
            base['grpo']['num_generations_per_prompt']=4;cfg.write_text(json.dumps(base))
            argv[-1]=str(root/'bad')
            with patch('sys.argv',argv),self.assertRaisesRegex(ValueError,'eight|=8'):
                main()
            self.assertFalse((root/'bad').exists())

if __name__=='__main__': unittest.main()
