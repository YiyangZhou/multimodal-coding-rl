"""CPU lifecycle checks; native CUDA acceptance is separate."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from mmrl.optimizer_offload import optimizer_cpu_during_backward

class Optimizer:
    def __init__(self): self.pre, self.post = [], []
    def register(self, hooks, callback):
        hooks.append(callback)
        return SimpleNamespace(remove=lambda: hooks.remove(callback))
    def register_step_pre_hook(self, callback): return self.register(self.pre, callback)
    def register_step_post_hook(self, callback): return self.register(self.post, callback)
    def step(self):
        for hook in self.pre: hook(self, (), {})
        for hook in self.post: hook(self, (), {})

class OffloadTest(unittest.TestCase):
    def setUp(self):
        self.moves = []
        self.worker = SimpleNamespace(optimizer=Optimizer(), cpu_offload=False,
                                      move_optimizer_to_device=self.moves.append)
        self.torch = patch.dict('sys.modules', {'torch': SimpleNamespace(
            cuda=SimpleNamespace(empty_cache=lambda: None))})
        self.torch.start()
        self.addCleanup(self.torch.stop)
    def assert_hooks_removed(self):
        self.assertEqual(self.worker.optimizer.pre + self.worker.optimizer.post, [])
    def test_multiple_optimizer_steps(self):
        with optimizer_cpu_during_backward(self.worker):
            self.assertEqual(self.moves, ['cpu'])
            self.worker.optimizer.step()
            self.worker.optimizer.step()
        self.assertEqual(self.moves, ['cpu', 'cuda', 'cpu', 'cuda', 'cpu', 'cuda'])
        self.assert_hooks_removed()
    def test_backward_failure(self):
        error = RuntimeError('backward failed')
        with self.assertRaises(RuntimeError) as caught:
            with optimizer_cpu_during_backward(self.worker): raise error
        self.assertIs(caught.exception, error)
        self.assertEqual(self.moves, ['cpu'])
        self.assert_hooks_removed()
    def test_transfer_failure_no_retry(self):
        def move(device):
            self.moves.append(device)
            if device == 'cuda': raise RuntimeError('transfer failed')
        self.worker.move_optimizer_to_device = move
        with self.assertRaisesRegex(RuntimeError, 'transfer failed'):
            with optimizer_cpu_during_backward(self.worker): self.worker.optimizer.step()
        self.assertEqual(self.moves, ['cpu', 'cuda'])
        self.assert_hooks_removed()
    def test_reject_incompatible_native_offload(self):
        self.worker.cpu_offload = True
        with self.assertRaises(ValueError):
            with optimizer_cpu_during_backward(self.worker): pass
        self.assertEqual(self.moves, [])
        self.assert_hooks_removed()

if __name__ == '__main__': unittest.main()
