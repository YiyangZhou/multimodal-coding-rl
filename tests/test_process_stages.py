import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from mmrl.process_stages import StageCollector
from mmrl.reward import InfrastructureError, Quality
from mmrl.service import CodingService
from mmrl.tasks import Task


class StageCollectorTests(unittest.IsolatedAsyncioTestCase):
    def collector(self, directory, sandbox, judge):
        return StageCollector(rollout_id='episode', task=Task('mmsvg_text2svg', 'task', 'Circle'),
                              route='codex', sandbox=sandbox, service=CodingService(sandbox, judge),
                              render_conditions={'image': 'pinned-test-image', 'judge': 'test-revision'},
                              audit_directory=directory)

    async def test_duplicate_source_uses_first_score_and_event_retry_is_idempotent(self):
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.read_artifact.return_value = '<svg/>'
        sandbox.render.return_value = b'independently-rendered'
        judge.grade.return_value = Quality(0.8, 0.6)
        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, sandbox, judge)
            a, b = await asyncio.gather(
                collector.capture_from_sandbox('handle', model_call_id='a', render_id='first'),
                collector.capture_from_sandbox('handle', model_call_id='b', render_id='second'))
            again = await collector.capture_from_sandbox('handle', model_call_id='a', render_id='first')
            self.assertEqual(a, again)
            self.assertEqual(a.reward, b.reward)
            self.assertEqual(len(collector.stages), 2)
            self.assertEqual(judge.grade.await_count, 1)
            self.assertEqual(sandbox.render.await_count, 1)
            records = [json.loads(p.read_text()) for p in Path(directory).glob('*/stage.json')]
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r['complete'] for r in records))
            self.assertEqual(sum('cached_verification' in r for r in records), 1)
            sandbox.read_artifact.return_value = '<svg>changed</svg>'
            self.assertEqual(a, await collector.capture_from_sandbox(
                'handle', model_call_id='a', render_id='first'))
            with self.assertRaises(InfrastructureError):
                await collector.capture_from_sandbox('handle', model_call_id='wrong', render_id='first')

    async def test_api_failure_is_unfinished_and_never_cached_as_zero(self):
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.read_artifact.return_value = '<svg/>'
        sandbox.render.return_value = b'independent-render'
        judge.grade.side_effect = InfrastructureError('Unavailable')
        with tempfile.TemporaryDirectory() as directory:
            collector = self.collector(directory, sandbox, judge)
            with self.assertRaises(InfrastructureError):
                await collector.capture_from_sandbox('handle', model_call_id='a', render_id='one')
            self.assertEqual(collector.stages, ())
            self.assertFalse(json.loads(next(Path(directory).glob('*/stage.json')).read_text())['complete'])
            sandbox.read_artifact.return_value = '<svg>different later version</svg>'
            with self.assertRaises(InfrastructureError):
                await collector.capture_from_sandbox('handle', model_call_id='a', render_id='one')
            self.assertEqual(judge.grade.await_count, 1)

    async def test_missing_artifact_is_execution_failure_without_api(self):
        sandbox, judge = AsyncMock(), AsyncMock()
        sandbox.read_artifact.side_effect = ValueError('No artifact')
        with tempfile.TemporaryDirectory() as directory:
            stage = await self.collector(directory, sandbox, judge).capture_from_sandbox(
                'handle', model_call_id='a', render_id='missing')
            self.assertEqual(stage.reward, 0)
            sandbox.render.assert_not_awaited()
            judge.grade.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
