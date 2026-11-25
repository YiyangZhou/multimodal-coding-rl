import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mmrl.audit import VerificationAudit, safe_terminal_metadata

class TerminalMetadataTests(unittest.TestCase):
    def test_persistence_and_reward_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = VerificationAudit(directory, task=SimpleNamespace(benchmark='fixture', task_id=1, prompt='existing prompt'), route='hermes', response_id='response-test', rollout_id='rollout-test', status='completed', terminal_metadata={'partial':'true', 'failed':'false', 'hermes_error':'SECRET', 'completed':'true', 'token':'SECRET'})
            result = {'reward':0.75, 'execution':1.0, 'mask_sample':False}
            audit.finish(result)
            saved = json.loads((audit.path/'record.json').read_text())
            self.assertEqual(saved['upstream_terminal'], {'partial':True, 'failed':False})
            self.assertNotIn('SECRET', json.dumps(saved))
            self.assertEqual(saved['result'], result)
            copied = json.loads((Path(directory)/'by-rollout/rollout-test'/(audit.path.name+'.json')).read_text())
            self.assertEqual(copied, saved)

    def test_absence_and_invalid_values_not_false(self):
        for value in (None, {}, {'partial':None}, {'completed':'true'}, {'partial':False}, {'failed':0}, {'partial':'TRUE'}, ['partial']):
            self.assertIsNone(safe_terminal_metadata('hermes', value))
        self.assertEqual(safe_terminal_metadata('hermes', {'failed':'false'}), {'failed':False})

    def test_unexposed_signals_omitted(self):
        for route in ('opencode','codex','unknown'):
            self.assertIsNone(safe_terminal_metadata(route, {'partial':'true','root_finish':'length','exitcode':0,'completed':'true'}))

if __name__=='__main__':
    unittest.main()
