import json
import tempfile
import unittest
from pathlib import Path
from mmrl.tasks import Task, sibling_mixed_row, write_rows
from mmrl.sibling_routes import apply_variant, ROUTES
from mmrl.mixed_balancing import balanced_indices

class MixedPreparationTest(unittest.TestCase):
    def test_nine_task_extensions_and_private_reference_boundary(self):
        for benchmark in ('manimbench_v1', '3dcodebench'):
            task=Task(benchmark=benchmark, task_id='1', prompt='Create the requested scene.', ground_truth='PRIVATE_REFERENCE_CODE')
            row=sibling_mixed_row(task)
            self.assertEqual(set(row['sibling_variants']), set(ROUTES))
            for index in range(8):
                resolved=apply_variant(dict(row, _rowidx=index),row['sibling_variants'])
                text=json.dumps(resolved['responses_create_params']['input'])
                self.assertNotIn('PRIVATE_REFERENCE_CODE', text)
                if index>=2:
                    self.assertIn('/opt/mmrl/render.py',text)
                    self.assertIn('/workspace/answer.py',text)
    def test_one_source_prompt_and_balanced_exposure(self):
        tasks=[Task('mmsvg_text2svg',str(i),'Draw a circle.') for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.jsonl'
            write_rows(tasks,p,'sibling-mixed',seed=42)
            rows=p.read_text().splitlines()
            self.assertEqual(len(rows),3)
            indices,report=balanced_indices(rows,benchmarks=['mmsvg_text2svg'])
            self.assertEqual(report['source_counts'],{'mmsvg_text2svg':3})
            self.assertEqual(len(indices),8)
    def test_single_only_benchmark_cannot_silently_route(self):
        task=Task('chartmimic','1','Draw.',references=['unused.png'],judge_references=['private.png'],category='direct')
        # Mock only image encoding: no need for actual reference assets here.
        from unittest.mock import patch
        with patch('mmrl.tasks.image_uri',return_value='data:image/png;base64,AA=='):
            with self.assertRaisesRegex(ValueError,'does not support route'):
                sibling_mixed_row(task)
if __name__=='__main__': unittest.main()
