"""Compare DP-global and local sequence padding through native loss and DeepEP.

Small-model checks, not full Super acceptance. All real tokens are retained.
"""
import argparse
import copy
from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_rl.models.automodel.local_padding import validate_local_padding_config
from nemo_rl.models.automodel import train
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossFn, ClippedPGLossConfig
from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def local(x):
    return x.to_local() if isinstance(x, DTensor) else x


def compare(a, b):
    a, b = local(a).float(), local(b).float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    if not a.count_nonzero():
        assert not b.count_nonzero()
        return {'relative_l2': 0., 'cosine': 1.}
    relative = float((a-b).norm() / a.norm())
    cosine = float(nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))
    assert relative < .02 and cosine > .9998, (relative, cosine)
    return {'relative_l2': relative, 'cosine': cosine}


class CausalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        h = self.proj(x).tanh().unsqueeze(1)
        return nn.functional.scaled_dot_product_attention(h, h, h, is_causal=True).squeeze(1) + x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1024, 64)
        self.body = CausalBlock()
        self.lm_head = nn.Linear(64, 1024, bias=False)

    def forward(self, input_ids, logits_to_keep=0, output_hidden_states=False):
        hidden = checkpoint(self.body, self.embed(input_ids), use_reentrant=False)
        selected = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        return CausalLMOutputWithPast(logits=self.lm_head(selected),
            hidden_states=(hidden,) if output_hidden_states else None)


def fetch_and_guards(cfg, rank):
    from nemo_rl.data_plane.adapters.local import LocalDataPlaneClient, local_batch_to_tensordict
    from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig
    from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN, DP_PADDING_TAG
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2Impl
    worker = object.__new__(DTensorPolicyWorkerV2Impl)
    worker.cfg = copy.deepcopy(cfg)
    worker._dp_client = LocalDataPlaneClient(LocalDataPlaneConfig())
    n = [17, 35, 23, 1][rank]
    fields = {'input_ids': torch.arange(n)[None, :], 'input_lengths': torch.tensor([n]),
              'token_mask': torch.ones(1,n), 'sample_mask': torch.ones(1)}
    worker._dp_client.register_partition('test', list(fields), 1, ['train'])
    meta = worker._dp_client.put_samples([f'row{rank}'], 'test',
        local_batch_to_tensordict(fields, batch_size=1), tags=[{DP_PADDING_TAG:rank==3}])
    meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = 35
    outputs = []
    for enabled in (False, True):
        worker.cfg['dtensor_cfg']['local_forward_padding'] = enabled
        validate_local_padding_config(worker.cfg)
        data = worker._fetch(meta, fetch_policy='independent')
        assert data['input_ids'].shape == (1, max(n, 2) if enabled else 35)
        assert torch.equal(data['input_ids'][:,:n], fields['input_ids'])
        assert data['sample_mask'].item() == (0 if rank == 3 else 1)
        assert data['token_mask'].sum() == (0 if rank == 3 else n)
        outputs.append(list(data['input_ids'].shape))
    assert fields['sample_mask'].item() == 1 and fields['token_mask'].sum() == n
    rejected = []
    for section, key, value in (
        ('dtensor_cfg','local_forward_padding','true'),
        ('dtensor_cfg','tensor_parallel_size',2), ('dtensor_cfg','context_parallel_size',2),
        ('dtensor_cfg','dp_replicate_size',2), ('dtensor_cfg','linear_logprob_chunk_size',None),
        ('sequence_packing','enabled',True), ('dynamic_batching','enabled',True),
        (None,'train_micro_batch_size',2), (None,'logprob_batch_size',2)):
        invalid = copy.deepcopy(worker.cfg)
        (invalid[section] if section else invalid)[key] = value
        try:
            validate_local_padding_config(invalid)
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError(f'Unsupported local padding accepted: {section}/{key}')
    return {'shapes':outputs,'masked_rank':3,'rejected':rejected}


def one_token_reproduction(cfg: dict) -> dict:
    from types import MethodType, SimpleNamespace
    from nemo_rl.data_plane.adapters.local import LocalDataPlaneClient, local_batch_to_tensordict
    from nemo_rl.data_plane.interfaces import LocalDataPlaneConfig
    from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import DTensorPolicyWorkerV2Impl
    from nemo_rl.models.automodel.linear_logprobs import model_next_token_logprobs
    from nemo_rl.models.automodel.data import process_microbatch
    worker = object.__new__(DTensorPolicyWorkerV2Impl)
    worker.cfg = copy.deepcopy(cfg)
    worker.cfg['dtensor_cfg']['local_forward_padding'] = True
    worker._dp_client = LocalDataPlaneClient(LocalDataPlaneConfig())
    fields = {'input_ids': torch.tensor([[0]]), 'input_lengths': torch.tensor([1]),
              'token_mask': torch.zeros(1,1), 'sample_mask': torch.zeros(1),
              'generation_logprobs': torch.zeros(1,1)}
    worker._dp_client.register_partition('placeholder', list(fields), 1, ['train'])
    meta = worker._dp_client.put_samples(['rejected'], 'placeholder',
        local_batch_to_tensordict(fields, batch_size=1), tags=[{}])
    meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] = 35
    worker._forward_pad_seqlen = MethodType(lambda self, meta: 0, worker)
    old = worker._fetch(meta, fetch_policy='independent')
    assert old['input_ids'].shape == (1,1)
    try:
        model_next_token_logprobs(None, {}, old['input_ids'], chunk_size=13, sampling_params=None)
    except ValueError as exc:
        assert str(exc) == 'Linear logprobs require a canonical [batch, sequence] token array'
    else:
        raise AssertionError('Original one-token failure did not reproduce')
    del worker._forward_pad_seqlen
    fixed = worker._fetch(meta, fetch_policy='independent')
    assert fixed['input_ids'].shape == (1,2)
    assert fixed['input_lengths'].item() == 1
    assert not fixed['sample_mask'].any() and not fixed['token_mask'].any()
    assert torch.equal(fixed['input_ids'][:,:1], fields['input_ids'])
    assert fields['input_ids'].shape == (1,1)
    processed = process_microbatch(fixed, SimpleNamespace(eos_token_id=0), False, worker.cfg)
    assert processed.input_ids.shape == (1,2)
    return {'old_error_reproduced':True, 'fixed_forward_shape':[1,2],
            'canonical_length':1,'zero_loss_masks':True,'source_unmodified':True}


def fsdp_phase(rank, mesh, raw, offload):
    cfg = copy.deepcopy(raw['policy'])
    cfg['dtensor_cfg'].update(linear_logprob_chunk_size=13,activation_cpu_offload=True,local_forward_padding=True)
    validate_local_padding_config(cfg)
    sampling = TrainingSamplingParams(temperature=.9)
    torch.manual_seed(19)
    models = [Model().cuda()]
    models.append(copy.deepcopy(models[0]))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32)
    for model in models:
        kwargs = dict(mesh=mesh, mp_policy=policy)
        if offload:
            kwargs['offload_policy'] = CPUOffloadPolicy(pin_memory=False)
        for child in (model.embed, model.body, model.lm_head):
            fully_shard(child, **kwargs)
        fully_shard(model, **kwargs)
    torch.manual_seed(127+rank)
    n = [17, 35, 23, 1][rank]
    ids = torch.randint(1024,(1,35),device='cuda')
    advantages = torch.randn(1,35,device='cuda')
    mask = torch.zeros(1,35,device='cuda'); mask[:,3:n] = 1; mask[:,::3] = 0
    sample_mask = torch.tensor([0. if rank==3 else 1.],device='cuda')
    valid_toks = (mask*sample_mask[:,None]).sum(); dist.all_reduce(valid_toks)
    records, probs, losses = [], [], []
    for model, size in zip(models,(35,max(n,2)),strict=True):
        data = BatchedDataDict({'input_ids':ids[:,:size], 'input_lengths':torch.tensor([n],device='cuda'),
            'advantages':advantages[:,:size], 'token_mask':mask[:,:size], 'sample_mask':sample_mask})
        processed = ProcessedMicrobatch(data,ProcessedInputs(data['input_ids'],size),1,size)
        prepared = train.PreparedModelForward({'input_ids':data['input_ids']},1,None,nullcontext)
        def run(post):
            return train.forward_with_post_processing_fn(model=model,prepared=prepared,
                post_processing_fn=post,processed_mb=processed,sampling_params=sampling,
                global_valid_seqs=torch.tensor(3.,device='cuda'),global_valid_toks=valid_toks)
        with torch.no_grad():
            lp,_,_ = run(train.LogprobsPostProcessor(cfg,False,sampling))
        probs.append(lp[:,:n])
        # Same fixed rollout/old-policy quantities for both layouts.
        old = probs[0]
        data['prev_logprobs'] = nn.functional.pad(old,(0,size-n)) + .01
        data['generation_logprobs'] = nn.functional.pad(old,(0,size-n)) - .03
        data['reference_policy_logprobs'] = nn.functional.pad(old,(0,size-n))
        fn = ClippedPGLossFn(ClippedPGLossConfig(**raw['loss_fn']))
        post = train.LossPostProcessor(fn,cfg,None,1,4,False,sampling)
        loss,_,_ = run(post); losses.append(float(loss.detach())); loss.backward()
        records.append(data)
    torch.testing.assert_close(probs[0],probs[1],rtol=1e-4,atol=.003)
    assert abs(losses[0]-losses[1]) < .0001,losses
    metrics = {}
    before = [local(p).detach().clone() for p in models[1].parameters()]
    for (name,a),(_,b) in zip(models[0].named_parameters(),models[1].named_parameters(),strict=True):
        metrics[name]=compare(a.grad,b.grad)
    for model in models:
        torch.optim.AdamW(model.parameters(),lr=5e-7,weight_decay=0).step()
    for (name,a),(_,b) in zip(models[0].named_parameters(),models[1].named_parameters(),strict=True):
        difference=float((local(a).detach()-local(b).detach()).abs().max())
        assert difference < 1.1e-6,(name,difference)
        metrics[name]['update_max_abs']=difference
    assert sum((local(p).detach()-old).count_nonzero() for p,old in zip(models[1].parameters(),before,strict=True))>0
    return {'cpu_offload':offload,'length':n,'global_length':35,'losses':losses,
            'logprob_max_abs':float((probs[0]-probs[1]).abs().max()),'gradients':metrics,'optimizer_updated':True}


def ep_phase(rank, mesh):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from nemo_automodel.components.models.common.utils import BackendConfig
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
    cfg = MoEConfig(n_routed_experts=8,n_shared_experts=0,n_activated_experts=2,
        n_expert_groups=1,n_limited_groups=1,train_gate=True,gate_bias_update_factor=0.,
        aux_loss_coeff=0.,score_func='softmax',route_scale=1.,dim=256,inter_dim=512,
        moe_inter_dim=512,norm_topk_prob=True,expert_activation='relu2',dtype=torch.bfloat16)
    models=[]
    for _ in range(2):
        torch.manual_seed(712)
        model=GroupedExpertsDeepEP(cfg,backend=BackendConfig(experts='torch_mm',dispatcher='deepep',expert_chunk_size=64)).cuda().to(torch.bfloat16)
        with torch.no_grad():
            model.gate_and_up_projs.normal_(0,.02);model.down_projs.normal_(0,.02)
        for name in ('gate_and_up_projs','down_projs'):
            setattr(model,name,nn.Parameter(distribute_tensor(getattr(model,name).detach(),mesh,[Shard(0)])))
        model.init_token_dispatcher(mesh);models.append(checkpoint_wrapper(model))
    torch.manual_seed(600+rank)
    n=[128,320,192,2][rank]
    hidden=torch.randn(320,256,device='cuda',dtype=torch.bfloat16)*.1
    probs,indices=torch.softmax(torch.randn(320,8,device='cuda'),-1).topk(2,-1)
    probs=probs/probs.sum(-1,keepdim=True)
    probe=torch.randn(320,256,device='cuda',dtype=torch.bfloat16)*.01
    outputs=[];inputs=[]
    for model,size in zip(models,(320,n),strict=True):
        x=hidden[:size].clone().requires_grad_();p=probs[:size].clone().requires_grad_()
        mask=torch.arange(size,device='cuda')<n
        y=model(x,mask,p,indices[:size]); ((y[:n]*probe[:n]).sum()*(0 if rank==3 else 1)).backward()
        outputs.append(y[:n].detach());inputs.append((x,p))
    torch.testing.assert_close(outputs[0],outputs[1],rtol=.02,atol=.002)
    errors=[compare(a.grad,b.grad) for a,b in zip(models[0].parameters(),models[1].parameters(),strict=True)]
    errors += [compare(a.grad[:n],b.grad) for a,b in zip(inputs[0],inputs[1],strict=True)]
    for model in models:
        torch.optim.AdamW(model.parameters(),lr=5e-7,weight_decay=0).step()
    for a,b in zip(models[0].parameters(),models[1].parameters(),strict=True):
        torch.testing.assert_close(local(a),local(b),rtol=1e-5,atol=1.1e-6)
    return {'passed':True,'length':n,'ep_size':4,'masked_local_loss_rank':3,'gradients':errors,'optimizer_compared':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy-config',type=Path,required=True)
    args=parser.parse_args();rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group(backend='cuda:nccl,cpu:gloo')
    try:
        assert dist.get_world_size()==4
        raw=json.loads(args.policy_config.read_text())
        cfg=copy.deepcopy(raw['policy']);cfg['dtensor_cfg']['linear_logprob_chunk_size']=13
        fetch=fetch_and_guards(cfg,rank)
        placeholder=one_token_reproduction(cfg)
        mesh=init_device_mesh('cuda',(4,),mesh_dim_names=('dp',))
        fsdp=[]
        for offload in (False,True):
            fsdp.append(fsdp_phase(rank,mesh,raw,offload));gc.collect();torch.cuda.empty_cache();dist.barrier()
        ep=ep_phase(rank,init_device_mesh('cuda',(4,),mesh_dim_names=('ep',)))
        args.output.mkdir(parents=True,exist_ok=True)
        result={'passed':True,'rank':rank,'fetch':fetch,'placeholder':placeholder,'fsdp':fsdp,'ep':ep,'full_super_update':False}
        (args.output/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'rank':rank,'passed':True}),flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
