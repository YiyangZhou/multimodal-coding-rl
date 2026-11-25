import copy
import unittest
from mmrl.tasks import Task, gym_row, reference_paths
from mmrl.webmmu_patch import apply_webmmu_patch
from mmrl.config import training_config

class AlignmentTests(unittest.TestCase):
    def test_four_same_extension_views_are_distinct_and_advertised(self):
        refs = [f'/assets/{name}.png' for name in ('front','back','left','right')]
        paths = reference_paths(refs)
        self.assertEqual(len(set(paths)), 4)
        task = Task('3dcodebench', 'four_views', 'Reconstruct', refs, ground_truth='reference code')
        row = gym_row(task, "hermes")
        import json
        text = json.dumps(row['responses_create_params']['input'])
        for path in paths:
            self.assertIn(path, text)
    def test_single_reference_keeps_existing_path(self):
        self.assertEqual(reference_paths(['/x/a.PNG']), ['/workspace/reference.png'])
        self.assertEqual(reference_paths([]), [])
        with self.assertRaises(ValueError): reference_paths(['/x/a.txt'])
    def test_fenced_position_only_insertion_uses_supplied_coordinates(self):
        source = '```html\n<html>\n<body>\n<p>original</p>\n</body>\n</html>\n```'
        result = apply_webmmu_patch(source, '@@ -3,0 +4 @@\n+<h1>Added</h1>')
        self.assertEqual(result, '<html>\n<body>\n<h1>Added</h1>\n<p>original</p>\n</body>\n</html>')
    def test_compose_preserves_explicit_dataset_and_balance(self):
        base = {'data_plane': {'enabled': True}, 'async_rl': {},
                'policy': {'generation': {'temperature': .8, 'top_p': .9, 'top_k': None}},
                'data': {'max_input_seq_length': 32768, 'train': {
                    'data_path': '/old', 'dataset_name': 'custom.BalancedDataset',
                    'benchmark_weights': {'manimbench_v1': 2}, 'seed': 123}},
                'grpo': {}, 'checkpointing': {}, 'logger': {}}
        original = copy.deepcopy(base)
        cfg = training_config(base, '/gym.json', '/new.jsonl', '/new-run')
        self.assertEqual(cfg['data']['train']['dataset_name'], 'custom.BalancedDataset')
        self.assertEqual(cfg['data']['train']['benchmark_weights'], {'manimbench_v1': 2})
        self.assertEqual(cfg['data']['train']['data_path'], '/new.jsonl')
        self.assertEqual(base, original)
