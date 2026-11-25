"""Optional Automodel worker that stages optimizer state around backward.

Set policy.worker_extension_cls_fqn to
mmrl.memory_worker.OptimizerOffloadPolicyWorker. Requires the pinned V2
DTensor worker and native GPU-resident optimizer. Does not change the loss.
"""
import ray
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2Impl
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from mmrl.optimizer_offload import optimizer_cpu_during_backward


class OptimizerOffloadPolicyWorkerImpl(DTensorPolicyWorkerV2Impl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        manager = self.checkpoint_manager
        if manager is None or manager.checkpointer is None:
            raise RuntimeError("Optimizer offload requires an initialized checkpoint manager")
        if manager.checkpointer.config.is_async:
            raise RuntimeError("Optimizer offload requires synchronous saving")

    def train(self, data, loss_fn, eval_mode=False, gbs=None, mbs=None,
              check_dim_skip_keys=None):
        if eval_mode:
            return super().train(data, loss_fn, eval_mode, gbs, mbs, check_dim_skip_keys)
        with optimizer_cpu_during_backward(self):
            return super().train(data, loss_fn, eval_mode, gbs, mbs, check_dim_skip_keys)


@ray.remote(runtime_env=get_runtime_env_for_policy_worker('dtensor_policy_worker_v2'))
class OptimizerOffloadPolicyWorker(OptimizerOffloadPolicyWorkerImpl):
    pass
