"""Keep Adam state off GPU during backward using native state transfers."""
from contextlib import contextmanager
import gc


@contextmanager
def optimizer_cpu_during_backward(worker):
    import torch
    optimizer = worker.optimizer
    if optimizer is None or worker.cpu_offload:
        raise ValueError('Requires the native GPU-resident Automodel optimizer')

    def before_step(optimizer, args, kwargs):
        worker.move_optimizer_to_device('cuda')

    def after_step(optimizer, args, kwargs):
        worker.move_optimizer_to_device('cpu')
        gc.collect()
        torch.cuda.empty_cache()

    pre = optimizer.register_step_pre_hook(before_step)
    post = optimizer.register_step_post_hook(after_step)
    try:
        worker.move_optimizer_to_device('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        yield
        # Preserve the original worker's residency contract between calls,
        # including checkpointing/refit. Native train already cleared grads.
        worker.move_optimizer_to_device('cuda')
    finally:
        pre.remove()
        post.remove()
        # On failure keep the original exception and leave state on CPU;
        # attempting a CUDA reload here could mask an OOM with a second OOM.
