from copy import deepcopy
from collections import Counter
import unittest
from mmrl.sibling_routes import apply_variant, validate_resolved_agent, mixed_metrics, ROUTES


class SiblingRoutesTest(unittest.TestCase):
    def setUp(self):
        self.task = {'task_id': 'task1', 'benchmark': 'chart2code', 'prompt': 'draw'}
        self.variants = {r: dict(task=self.task, route=r,
            agent_ref={'type': 'responses_api_agents', 'name': 'mm_' + r},
            responses_create_params={'input': [{'role': 'user', 'content': r}],
                                     'max_output_tokens': 32768}) for r in ROUTES}
        self.row = dict(task=self.task, _rowidx=0, _ng_rollout_id='receipt0',
                        group_id='shared-group', attempt=2,
                        responses_create_params={'input': [], 'max_output_tokens': 16384,
                                                 'temperature': 1.0, 'top_p': 1.0})

    def test_group_distribution_and_receipts(self):
        rows = [apply_variant(dict(self.row, _rowidx=i, _ng_rollout_id=f'receipt{i}'), self.variants)
                for i in range(8)]
        self.assertEqual(Counter(r['route'] for r in rows), {r: 2 for r in ROUTES})
        self.assertEqual({r['group_id'] for r in rows}, {'shared-group'})
        self.assertEqual(len({r['_ng_rollout_id'] for r in rows}), 8)
        self.assertTrue(all(r['responses_create_params']['max_output_tokens'] == 16384 for r in rows))

    def test_partial_resume_does_not_renumber_routes(self):
        for i in [7, 2, 5]:
            row = dict(self.row, _rowidx=i)
            self.assertEqual(apply_variant(row, self.variants)['route'], ROUTES[i // 2])

    def test_resolved_agent_must_match_before_durable_completion(self):
        row = apply_variant(dict(self.row, _rowidx=4), self.variants)
        validate_resolved_agent(row, {'type':'responses_api_agents','name':'mm_codex'})
        before = deepcopy(row)
        with self.assertRaises(ValueError):
            validate_resolved_agent(row, {'type':'responses_api_agents','name':'mm_hermes'})
        self.assertEqual(row, before)
        validate_resolved_agent({'agent_ref':{'name':'legacy'}}, {'name':'resolved_alias'})

    def test_metrics_preserve_global_mean_and_route_assignment(self):
        rows = [apply_variant(dict(self.row, _rowidx=i), self.variants) for i in range(8)]
        def compute(values, agent):
            mean = sum(values)/len(values)
            return {'total_reward/mean':mean, agent+'/reward/mean':mean}
        metrics = mixed_metrics(list(range(8)), list(reversed(rows)), compute,
                                lambda row: row['route'])
        self.assertEqual(metrics['total_reward/mean'], 3.5)
        for route, mean in zip(ROUTES,[.5,2.5,4.5,6.5]):
            self.assertEqual(metrics[route+'/reward/mean'],mean)
            self.assertEqual(metrics[route+'/sibling_share'],.25)
        partial = mixed_metrics([2,7], [rows[7],rows[2]], compute,lambda r:r['route'])
        self.assertEqual(partial['opencode/reward/mean'],2)
        self.assertEqual(partial['hermes/reward/mean'],7)

    def test_wrong_task_rejected(self):
        bad = deepcopy(self.variants); bad['codex']['task']['task_id'] = 'other'
        with self.assertRaises(ValueError): apply_variant(self.row, bad)

    def test_no_mutation(self):
        before = deepcopy((self.row, self.variants))
        apply_variant(self.row, self.variants)
        self.assertEqual(before, (self.row, self.variants))


if __name__ == '__main__':
    unittest.main()
