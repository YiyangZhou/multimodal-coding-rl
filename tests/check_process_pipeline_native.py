# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real CPU TransferQueue -> capture finalizer -> native advantage stage.

Uses synthetic captured calls, without generation, sandbox or optimizer update.
"""
import argparse
import asyncio
from dataclasses import asdict, replace
import json
import os
import select
import subprocess
from pathlib import Path

import ray
import torch

from mmrl.process_rewards import ProcessRewardConfig
from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
from nemo_rl.algorithms.single_controller_utils.advantage_stage import AdvantageComputer, AdvantageRequest, AdvantageStageConfig
from nemo_rl.algorithms.single_controller_utils.utils import tensor_field, fields_for_put
from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.data_plane.schema import SC_ROLLOUT_SCHEMA_FIELDS
from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSink
from nemo_rl.experience.process_returns import ProcessReturnConfig
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.data_plane.token_capture_test_fixtures import build_fixture_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-config', type=Path, required=True)
    parser.add_argument('--gym-python', type=Path, required=True,
                        help='Locked Gym interpreter; the learner need not install Gym server dependencies')
    args = parser.parse_args()
    raw = json.loads(args.policy_config.read_text())
    hooks = ProcessReturnConfig(binder='mmrl.process_capture.bind_native_returns',
                               advantage='mmrl.process_advantages.compute_process_advantages')
    raw['token_capture']['process_returns'] = hooks.model_dump()
    raw['grpo']['adv_estimator']['normalize_rewards'] = False
    raw['grpo']['adv_estimator']['use_leave_one_out_baseline'] = True
    raw['async_rl']['log_full_train_data'] = False
    config = MasterConfig(**raw)
    stage_config = AdvantageStageConfig.from_master_config(config)
    assert stage_config.process_returns == hooks and 'process_returns' in stage_config.input_fields()
    ray.init(num_cpus=4, include_dashboard=False, object_store_memory=256 * 1024**2,
             _temp_dir=f'/tmp/mmrl-returns-{os.getpid()}')
    client = build_data_plane_client({'enabled': True, 'impl': 'transfer_queue',
        'backend': 'simple', 'claim_meta_poll_interval_s': .5,
        'simple': {'storage_capacity': 128, 'num_storage_units': 1}})
    results = []
    worker = subprocess.Popen(
        [str(args.gym_python), '-B', str(Path(__file__).with_name('check_harness_verifier_wire.py')),
         '--transport-stream'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def through_harness(verifier: dict) -> dict:
        worker.stdin.write(json.dumps(verifier) + '\n')
        worker.stdin.flush()
        if not select.select([worker.stdout], [], [], 180)[0]:
            raise TimeoutError('Gym HTTP verifier test did not respond within 180 seconds')
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError('Gym HTTP verifier test exited without a response')
        return json.loads(line)

    try:
        client.register_partition(partition_id='staging', fields=list(STAGING_FIELDS),
                                  num_samples=128, consumer_tasks=['finalize'])
        client.register_partition(partition_id='canonical', fields=list(SC_ROLLOUT_SCHEMA_FIELDS),
                                  num_samples=128, consumer_tasks=['train'])
        sink = TQTokenSink(client, staging_partition='staging')
        for case in ('mixed', 'bad_identity', 'truncated', 'logprob_error', 'disabled'):
            enabled = case != 'disabled'
            finalizer = RolloutReassembler(client, partition_id='canonical', staging_partition='staging',
                pad_token_id=0, max_seq_len=7 if case == 'truncated' else 4096,
                process_returns=hooks if enabled else None)
            receipts, rows, rollout_ids = [], [], []
            for index, score in enumerate((.8, .4, 0.)):
                rid = f'{case}_g{index}'
                records, receipt, row = build_fixture_artifacts(
                    'worked_example' if index == 0 else 'single_call', rollout_id=rid)
                for record in records:
                    assert sink.stage(record).ok
                wire = receipt.model_dump()
                if enabled and index != 2:
                    coefficients = ProcessRewardConfig() if index == 0 else ProcessRewardConfig(
                        progress_weight=0, token_cost_per_1000=0, render_cost=0, duplicate_render_cost=0)
                    stages = [{'rollout_id': rid, 'model_call_id': 'c1', 'render_id': 'first',
                               'reward': .5, 'source_signature': 'source',
                               'render_conditions_signature': 'view'}] if index == 0 else []
                    wire['process_reward_payload'] = {'version': 1, 'rollout_id': rid,
                        'final_reward': score, 'stages': stages, 'config': asdict(coefficients)}
                    if case == 'bad_identity' and index == 0:
                        wire['process_reward_payload']['stages'][0]['model_call_id'] = 'discarded-branch'
                # Exercise actual HTTP response serialization before attaching
                # verifier metadata to the receipt, as the native Gym adapter does.
                verifier = {'reward': score, 'mask_sample': index == 2,
                            'instance_config': {'mask_sample': index == 2}}
                if 'process_reward_payload' in wire:
                    verifier['process_reward_payload'] = wire.pop('process_reward_payload')
                transported = through_harness(verifier)
                assert transported['mask_sample'] is (index == 2)
                assert transported['instance_config']['mask_sample'] is (index == 2)
                if 'process_reward_payload' in verifier:
                    assert transported.get('process_reward_payload') == verifier['process_reward_payload']
                    wire['process_reward_payload'] = transported['process_reward_payload']
                receipts.append(wire)
                rows.append(row)
                rollout_ids.append(rid)
            # A failed infrastructure row intentionally has no process payload.
            group = finalizer.finalize_group(case, rollout_ids, receipts, [.8, .4, 0.],
                mask_sample=[False, False, True], fallback_weight_version=4, prompt_idx=0)
            meta = group.meta
            assert meta is not None and len(meta.sample_ids) == 3
            fields = ['input_ids', 'token_mask', 'sample_mask', 'mask_sample',
                      'truncated', 'total_reward', 'generation_logprobs']
            if enabled:
                fields.append('process_returns')
            data = client.get_samples(meta.sample_ids, 'canonical', select_fields=fields)
            generation = tensor_field(data, 'generation_logprobs')
            policy = generation.clone()
            if case == 'logprob_error':
                policy[0] += 10
            client.put_samples(meta.sample_ids, 'canonical', fields=fields_for_put(meta, {'prev_logprobs': policy}))
            cfg = stage_config if enabled else replace(stage_config, process_returns=None)
            estimator = GRPOAdvantageEstimator(config.grpo.adv_estimator, config.loss_fn)
            computer = AdvantageComputer(client, config=cfg, advantage_estimator=estimator)
            asyncio.run(computer.run(AdvantageRequest(meta=meta)))
            final = client.get_samples(meta.sample_ids, 'canonical', select_fields=['advantages', 'sample_mask'])
            advantages = tensor_field(final, 'advantages')
            valid = tensor_field(final, 'sample_mask').reshape(-1)
            token_mask = tensor_field(data, 'token_mask')
            assert not (advantages[2] * valid[2]).any()
            if case in ('bad_identity', 'truncated', 'logprob_error'):
                assert not (advantages * valid[:, None] * token_mask).any(), case
                assert valid[0] == 0 and valid[1] == 1
            elif case == 'mixed':
                totals = tensor_field(data, 'total_reward').reshape(-1)
                returns = tensor_field(data, 'process_returns')
                assert returns.shape == token_mask.shape
                assert not returns[~token_mask.bool()].any()
                assert returns[0, 2] != returns[0, 5], 'Per-call rewards disappeared in transport'
                expected = torch.zeros_like(advantages)
                expected[0] = (returns[0] - totals[1]) * token_mask[0]
                expected[1] = (returns[1] - totals[0]) * token_mask[1]
                torch.testing.assert_close(advantages, expected, atol=1e-7, rtol=0)
                assert float(totals[1]) == float(torch.tensor(.4))
            else:
                expected = torch.tensor([.4, -.4, 0.])[:, None] * token_mask * valid[:, None]
                torch.testing.assert_close(advantages * token_mask * valid[:, None], expected, atol=1e-7, rtol=0)
            results.append({'case': case, 'passed': True, 'valid_after_filters': valid.tolist(),
                            'valid_rows_at_finalization': group.valid_row_count})
        assert not torch.cuda.is_initialized()
        with args.output.open('x') as stream:
            json.dump({'passed': True, 'real_transfer_queue': True, 'cases': results,
                       'native_harness_http_verifier_transport': True,
                       'gym_python': str(args.gym_python),
                       'full_model_update': False, 'real_harness_rollout': False}, stream, indent=2)
        print(json.dumps(results), flush=True)
    finally:
        worker.stdin.close()
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        worker.stdout.close()
        client.close()
        ray.shutdown()


if __name__ == '__main__':
    main()
