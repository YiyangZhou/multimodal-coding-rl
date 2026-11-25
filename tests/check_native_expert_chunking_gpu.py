"""Four-GPU parity through FSDP/checkpointing and real DeepEP dispatch.

These small component updates do not replace full-model checkpoint acceptance.
Run with torchrun --standalone --nproc_per_node=4 and --output DIRECTORY.
"""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP, _torch_mm_experts_fwd, relu2_deepep


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def compare(left, right):
    a, b = local(left).float(), local(right).float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    scale = float(a.norm())
    if scale == 0:
        assert torch.equal(a, b)
        return 0.0
    relative = float((a-b).norm()) / scale
    cosine = float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    assert relative < .02 and cosine > .9999, (relative, cosine)
    return relative


class LocalExperts(torch.nn.Module):
    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.up = torch.nn.Parameter(torch.randn(4, 256, 512) * .02)
        self.down = torch.nn.Parameter(torch.randn(4, 512, 256) * .02)

    def forward(self, hidden, probs, counts):
        return _torch_mm_experts_fwd(hidden, self.up, self.down, counts, probs,
            relu2_deepep, max_chunk_rows=self.chunk_size)


def fsdp_phase(rank, mesh):
    models = []
    for chunk in (0, 64):
        torch.manual_seed(183)
        model = checkpoint_wrapper(LocalExperts(chunk).cuda())
        fully_shard(model, mesh=mesh, reshard_after_forward=True,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
        models.append(model)
    torch.manual_seed(250 + rank)
    n = 256 + rank * 64
    counts = torch.tensor([64, 64, 0, n-128], device="cuda")
    hidden = torch.randn(n, 256, device="cuda", dtype=torch.bfloat16) * .1
    probs = torch.rand(n, 1, device="cuda")
    probe = torch.randn(n, 256, device="cuda", dtype=torch.bfloat16) * .01
    outputs, inputs = [], []
    optimizers = [torch.optim.AdamW(m.parameters(), lr=5e-7) for m in models]
    for model in models:
        x, p = hidden.clone().requires_grad_(), probs.clone().requires_grad_()
        y = model(x, p, counts)
        # Keep a fully masked rank in the collective/checkpoint graph.
        loss = (y * probe).sum() * (0 if rank == 3 else 1)
        loss.backward()
        outputs.append(y.detach());inputs.append((x,p))
    torch.testing.assert_close(outputs[0],outputs[1],rtol=.02,atol=.002)
    errors = [compare(a.grad,b.grad) for a,b in zip(models[0].parameters(),models[1].parameters(),strict=True)]
    errors += [compare(a.grad,b.grad) for a,b in zip(inputs[0],inputs[1],strict=True)]
    local_gradient_nonzeros = sum(torch.count_nonzero(local(p.grad)).item() for p in models[1].parameters())
    before = [local(p).detach().clone() for p in models[1].parameters()]
    for opt in optimizers: opt.step()
    for a,b in zip(models[0].parameters(),models[1].parameters(),strict=True):
        torch.testing.assert_close(local(a),local(b),atol=1e-6,rtol=1e-5)
    local_update_nonzeros = sum(
        torch.count_nonzero(local(p) - old).item()
        for p, old in zip(models[1].parameters(), before, strict=True)
    )
    # FSDP shards the four experts along dimension 0. Expert 2 receives no
    # rows on ANY rank; it must not be required to have a gradient/update.
    # Rank 3 has a masked local loss, but its expert receives reduced gradients
    # from the other ranks and must still update.
    metrics = [None] * dist.get_world_size()
    dist.all_gather_object(metrics, {
        "rank": rank, "gradient_nonzeros": local_gradient_nonzeros,
        "update_nonzeros": local_update_nonzeros,
    })
    for metric in metrics:
        if metric["rank"] == 2:
            assert metric["gradient_nonzeros"] == 0, metrics
            assert metric["update_nonzeros"] == 0, metrics
        else:
            assert metric["gradient_nonzeros"] > 0, metrics
            assert metric["update_nonzeros"] > 0, metrics
    return {"passed":True,"max_gradient_relative_l2":max(errors),
        "nonzero_optimizer_update":True,"masked_rank":3,
        "inactive_expert_rank":2,"rank_update_metrics":metrics}


def ep_phase(rank, mesh):
    config = MoEConfig(n_routed_experts=8,n_shared_experts=0,n_activated_experts=2,
        n_expert_groups=1,n_limited_groups=1,train_gate=True,gate_bias_update_factor=0.,
        aux_loss_coeff=0.,score_func="softmax",route_scale=1.,dim=256,inter_dim=512,
        moe_inter_dim=512,norm_topk_prob=True,expert_activation="relu2",dtype=torch.bfloat16)
    models=[]
    for chunk in (0,64):
        backend=BackendConfig(experts="torch_mm",dispatcher="deepep",expert_chunk_size=chunk)
        torch.manual_seed(712)
        model=GroupedExpertsDeepEP(config,backend=backend).cuda().to(torch.bfloat16)
        with torch.no_grad():
            model.gate_and_up_projs.normal_(0,.02);model.down_projs.normal_(0,.02)
        for name in ("gate_and_up_projs","down_projs"):
            parameter=getattr(model,name)
            setattr(model,name,torch.nn.Parameter(distribute_tensor(parameter.detach(),mesh,[Shard(0)])))
        model.init_token_dispatcher(mesh)
        models.append(checkpoint_wrapper(model))
    torch.manual_seed(600+rank)
    n=128+64*rank
    hidden=torch.randn(n,256,device="cuda",dtype=torch.bfloat16)*.1
    probabilities,indices=torch.softmax(torch.randn(n,8,device="cuda"),-1).topk(2,-1)
    probabilities=probabilities/probabilities.sum(-1,keepdim=True)
    mask=torch.ones(n,device="cuda",dtype=torch.bool);mask[-1]=False
    probe=torch.randn(n,256,device="cuda",dtype=torch.bfloat16)*.01
    outputs=[];inputs=[]
    for model in models:
        x=hidden.clone().requires_grad_();p=probabilities.clone().requires_grad_()
        y=model(x,mask,p,indices);(y*probe).sum().backward()
        outputs.append(y.detach());inputs.append((x,p))
    torch.testing.assert_close(outputs[0],outputs[1],rtol=.02,atol=.002)
    errors=[compare(a.grad,b.grad) for a,b in zip(models[0].parameters(),models[1].parameters(),strict=True)]
    errors += [compare(a.grad,b.grad) for a,b in zip(inputs[0],inputs[1],strict=True)]
    return {"passed":True,"max_gradient_relative_l2":max(errors),"ep_size":4,"uneven_rank_rows":True,"outer_activation_checkpointing":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group(backend='cuda:nccl,cpu:gloo')
    assert dist.get_world_size()==4
    mesh=init_device_mesh('cuda',(4,),mesh_dim_names=('dp',))
    fsdp=fsdp_phase(rank,mesh);gc.collect();torch.cuda.empty_cache();dist.barrier()
    ep_mesh=init_device_mesh('cuda',(4,),mesh_dim_names=('ep',))
    ep=ep_phase(rank,ep_mesh)
    args.output.mkdir(parents=True,exist_ok=True)
    report={'rank':rank,'passed':True,'fsdp':fsdp,'ep':ep,'full_model_acceptance':False}
    (args.output/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    main()
