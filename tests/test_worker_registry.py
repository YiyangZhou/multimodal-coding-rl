import unittest
from mmrl.worker_registry import MEMORY_WORKER, NATIVE_WORKER, register_memory_worker

class RegistryTest(unittest.TestCase):
    def test_native_environment_preserved_and_idempotent(self):
        registry = {NATIVE_WORKER: '/locked/python'}
        config = {'policy': {'worker_extension_cls_fqn': MEMORY_WORKER}}
        register_memory_worker(config, registry)
        register_memory_worker(config, registry)
        self.assertEqual(registry[MEMORY_WORKER], '/locked/python')
    def test_conflicting_environment_rejected(self):
        registry = {NATIVE_WORKER: '/locked/python', MEMORY_WORKER: '/wrong/python'}
        with self.assertRaises(ValueError):
            register_memory_worker({'policy': {'worker_extension_cls_fqn': MEMORY_WORKER}}, registry)
    def test_unrelated_worker_untouched(self):
        registry = {}
        register_memory_worker({'policy': {'worker_extension_cls_fqn': 'other.Worker'}}, registry)
        self.assertEqual(registry, {})
    def test_missing_native_environment_rejected(self):
        with self.assertRaises(KeyError):
            register_memory_worker({'policy': {'worker_extension_cls_fqn': MEMORY_WORKER}}, {})

if __name__ == '__main__': unittest.main()
