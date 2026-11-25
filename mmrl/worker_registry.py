"""Bind external workers to their native dependency environment."""
MEMORY_WORKER = 'mmrl.memory_worker.OptimizerOffloadPolicyWorker'
NATIVE_WORKER = 'nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2'


def register_memory_worker(config, registry=None):
    if config.get('policy', {}).get('worker_extension_cls_fqn') != MEMORY_WORKER:
        return
    if registry is None:
        from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
        registry = ACTOR_ENVIRONMENT_REGISTRY
    native = registry[NATIVE_WORKER]
    existing = registry.get(MEMORY_WORKER, native)
    if existing != native:
        raise ValueError('Optimizer-offload worker must use the native DTensor V2 environment')
    registry[MEMORY_WORKER] = native
