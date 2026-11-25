"""Diagnostics preserve failure boundaries without persisting exception payloads."""
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mmrl.audit import VerificationAudit, safe_failure_markers
from mmrl.catalog import BENCHMARKS
from mmrl.reward import InfrastructureError
from mmrl.sandbox import RemoteSandbox
from mmrl.tasks import Task


class SafeDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def test_markers_never_persist_raw_exception(self):
        message = 'SandboxAPIException failed to run command status code: 502 https://private.invalid/?token=secret'
        self.assertEqual(safe_failure_markers(message), ['sandbox_http_502'])
        with tempfile.TemporaryDirectory() as directory:
            audit = VerificationAudit(directory, task=Task('mmsvg_text2svg', 'x', 'Circle'),
                                      route='hermes', response_id='r', status='failed', failure_code=message)
            audit.finish({'failure_reason': 'artifact_symlink'})
            text = (audit.path / 'record.json').read_text()
            self.assertNotIn('secret', text)
            self.assertNotIn('private.invalid', text)
            self.assertEqual(json.loads(text)['result']['artifact_failure_code'], 'artifact_symlink')

    async def test_artifact_failures_do_not_download(self):
        sandbox = object.__new__(RemoteSandbox)
        sandbox.provider = AsyncMock()
        name = next(iter(BENCHMARKS.values())).artifact
        for code, reason in [(41, 'artifact_missing_or_not_regular'), (42, 'artifact_symlink'),
                             (43, 'artifact_oversize'), (44, 'artifact_size_probe_failed'),
                             (1, 'artifact_probe_failed')]:
            sandbox.provider.exec.return_value = SimpleNamespace(return_code=code, error_type=None)
            with self.assertRaisesRegex(ValueError, '^' + reason + '$'):
                await sandbox.read_artifact('handle', name)
        sandbox.provider.exec.return_value = SimpleNamespace(return_code=42, error_type='transport')
        with self.assertRaises(InfrastructureError):
            await sandbox.read_artifact('handle', name)
        sandbox.provider.download_file.assert_not_awaited()
