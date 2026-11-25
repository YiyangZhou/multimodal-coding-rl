"""CPU behavioral checks of patched source, without importing GPU runtimes.

python tests/check_super_multiturn.py --rl-root /path/to/rl --gym-root /path/to/gym
Uses extracted production methods with recording transport/backend substitutes.
"""
import argparse
import ast
import asyncio
import inspect
import logging
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Optional


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name in names]
    assert len(nodes) == len(names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


async def check_snapshot(root):
    env = extract(root / 'nemo_rl/algorithms/single_controller.py',
                  {'_commit_snapshot_before_cancellation'}, {'asyncio': asyncio})
    commit = env['_commit_snapshot_before_cancellation']
    assert await commit(lambda: 7) == 7
    def fail():
        raise OSError('controlled failure')
    try:
        await commit(fail)
    except OSError:
        pass
    else:
        raise AssertionError('commit error swallowed')
    with tempfile.TemporaryDirectory() as directory:
        tmp, final = Path(directory) / 'tmp', Path(directory) / 'final'
        tmp.mkdir()
        (tmp / 'manifest').write_text('committed')
        started, release = threading.Event(), threading.Event()
        def rename():
            started.set()
            assert release.wait(5)
            tmp.rename(final)
        task = asyncio.create_task(commit(rename))
        while not started.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.02)
        assert not task.done() and tmp.exists()
        release.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('cancellation swallowed')
        assert (final / 'manifest').read_text() == 'committed' and not tmp.exists()


def check_registration(root):
    class Queue:
        def __init__(self, fields, fail=False):
            self.fields, self.fail, self.puts = fields, fail, []
        def get_client(self):
            return self
        def get_partition_list(self):
            return ['p'] if self.fields is not None else []
        def get_meta(self, **kwargs):
            assert kwargs['mode'] == 'force_fetch' and kwargs['batch_size'] == 0
            if self.fail:
                raise RuntimeError('transport unavailable')
            return SimpleNamespace(field_schema={f: {} for f in self.fields})
        def kv_batch_put(self, **kwargs):
            assert not set(kwargs['fields']).intersection(self.fields or [])
            self.puts.append(set(kwargs['fields']))
        def kv_clear(self, **kwargs):
            pass
    for existing, wanted, expected in [(['tokens', 'mask'], ['tokens', 'mask'], []),
                                       (['tokens'], ['tokens', 'new'], [{'new'}]),
                                       (None, ['tokens'], [{'tokens'}]),
                                       ([], ['tokens'], [{'tokens'}])]:
        queue = Queue(existing)
        env = extract(root / 'nemo_rl/data_plane/adapters/transfer_queue.py', {'register_partition'},
                      {'tq': queue, 'torch': SimpleNamespace(zeros=lambda n: 0.0),
                       'TensorDict': lambda data, **kw: data, 'os': os, 'time': time})
        obj = SimpleNamespace(_warmed_fields={}, _mark_data_operation_started=lambda: None)
        for _ in range(2):
            env['register_partition'](obj, partition_id='p', fields=wanted,
                                      num_samples=8, consumer_tasks=['train'])
            assert queue.puts == expected
    queue = Queue(['tokens'], fail=True)
    env['tq'] = queue
    obj = SimpleNamespace(_warmed_fields={}, _mark_data_operation_started=lambda: None)
    try:
        env['register_partition'](obj, partition_id='p', fields=['tokens'], num_samples=8, consumer_tasks=['train'])
    except RuntimeError:
        pass
    else:
        raise AssertionError('transport error swallowed')
    assert not queue.puts


def check_context_boundary(root):
    class ValidationError(Exception):
        def __init__(self, message, **kwargs):
            super().__init__(message)
            self.__dict__.update(kwargs)
    env = extract(root / 'nemo_rl/models/generation/vllm/vllm_worker_async.py',
                  {'_clamp_max_tokens', '_set_max_tokens'},
                  {'Optional': Optional, 'LOGGER': logging.getLogger('check'), 'VLLMValidationError': ValidationError})
    probe_type = type('Probe', (), {name: env[name] for name in ('_clamp_max_tokens', '_set_max_tokens')})
    probe = probe_type()
    probe.model_config = SimpleNamespace(max_model_len=32)
    for budget in (None, 1, 100):
        for length in (32, 33):
            request = SimpleNamespace(max_tokens=budget, max_completion_tokens=None)
            try:
                probe._clamp_max_tokens(request, budget, [0] * length)
            except ValidationError as error:
                assert error.parameter == 'input_tokens'
            else:
                raise AssertionError('context overflow accepted')
    request = SimpleNamespace(max_tokens=100, max_completion_tokens=None)
    probe._clamp_max_tokens(request, 100, [0] * 30)
    assert request.max_tokens == 2
    request = SimpleNamespace(max_tokens=None, max_completion_tokens=None)
    probe._clamp_max_tokens(request, None, [0])
    assert request.max_tokens is None


async def check_poisoned_dispatch(root):
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code, self.detail = status_code, detail
    state = {}
    async def resolve(messages):
        state['resolved'] += 1
    async def intent():
        state['intents'] += 1
    async def capture(result, **kwargs):
        state['captured'] += 1
    async def backend(body):
        state['backend'] += 1
        return {'id': 'unchanged-answer'}
    env = extract(root / 'nemo_gym/base_responses_api_model.py',
                  {'_invoke_chat_completions', '_invoke_responses', '_reject_poisoned_rollout', 'RolloutCaptureFailed'},
                  dict(inspect=inspect, HTTPException=HTTPException,
                       current_capture_context=lambda: state['context'], _request_messages=lambda x: [],
                       resolve_parent=resolve, register_call_intent=intent, capture_tokens=capture,
                       UNCOMMITTED_CALL_REASON='request_finished_without_staged_coordinates'))
    for route in ('_invoke_chat_completions', '_invoke_responses'):
        for reason in (None, 'unresolved_parent', 'worker_capture_failed', 'request_finished_without_staged_coordinates'):
            async def manifest(rollout):
                return {'failures': [] if reason is None else [{'reason': reason}]}
            state.update(context=SimpleNamespace(external_staging=True, rollout_id='r',
                                                 lineage_store=SimpleNamespace(manifest=manifest)),
                         resolved=0, intents=0, captured=0, backend=0)
            server = SimpleNamespace(chat_completions=backend, responses=backend)
            try:
                result = await env[route](server, None, object())
            except HTTPException as error:
                assert reason == 'unresolved_parent' and error.status_code == 400
                assert state['backend'] == state['intents'] == state['captured'] == 0
            else:
                assert reason != 'unresolved_parent' and result == {'id': 'unchanged-answer'}
                assert state['backend'] == state['intents'] == state['captured'] == 1
        for context in (None, SimpleNamespace(external_staging=False)):
            state.update(context=context, resolved=0, intents=0, captured=0, backend=0)
            assert await env[route](server, None, object()) == {'id': 'unchanged-answer'}
            assert state['backend'] == 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-root', type=Path, required=True)
    parser.add_argument('--gym-root', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(check_snapshot(args.rl_root))
    check_registration(args.rl_root)
    check_context_boundary(args.rl_root)
    asyncio.run(check_poisoned_dispatch(args.gym_root))
    print('Passed: snapshot cancellation, restored queue schemas, context boundary, poisoned-rollout dispatch')
