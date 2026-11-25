"""Opt-in native evaluation services; no credentials, deployment or training hooks.

Call evaluation_classes(ledger) before wrapping the returned resource class with
benchmark-specific seed_session overrides. Requires the pinned Gym/VLLM runtime.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import re

class BudgetTerminal(Exception):
    """A locally proven budget limit, never an upstream exception category."""

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)

class BudgetLedger:
    def __init__(self, root, per_response=96000, total=192000, context=262144):
        self.root, self.per_response, self.total = Path(root), per_response, total
        self.rows, self.locks = {}, {}
        self.context = context

    def register(self, rollout, sample):
        assert re.fullmatch(r'[a-f0-9]{32}', rollout) and rollout not in self.rows
        self.rows[rollout] = dict(sample_id=sample, rollout_id=rollout, output_tokens=0,
                                  calls=0, truncated=False, infrastructure_error=False)
        self.locks[rollout] = asyncio.Lock()
        self.commit(rollout)

    def commit(self, rollout):
        save(self.root / rollout / 'budget.json', self.rows[rollout])

    def allowance(self, rollout):
        row = self.rows[rollout]
        if row['infrastructure_error']:
            raise ValueError('Previous upstream failure requires a fresh attempt')
        remaining = self.total - row['output_tokens']
        if remaining <= 0:
            row['truncated'] = True
            self.commit(rollout)
            raise ValueError('Per-task output budget exhausted')
        return min(self.per_response, remaining)

    def charge(self, rollout, response):
        row = self.rows[rollout]
        usage = response['usage']['completion_tokens']
        if type(usage) is not int or usage < 0 or usage > min(self.allowance(rollout), row.get('last_allowance', self.per_response)):
            raise ValueError('Missing or inconsistent completion token usage')
        row['output_tokens'] += usage
        row['calls'] += 1
        row['truncated'] |= any(c.get('finish_reason') == 'length' for c in response['choices'])
        self.commit(rollout)

def evaluation_classes(ledger):
    from fastapi import HTTPException, Request
    from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
    from mmrl.gym_app import VerifyRequest
    from pydantic import PrivateAttr
    from nemo_gym.rollout_correlation import current_rollout_id
    from responses_api_models.vllm_model.app import VLLMModel
    from mmrl.gym_app import CodingResourcesServer, VerifyResponse
    from mmrl.sandbox import RemoteSandbox
    from mmrl.tasks import Task
    from mmrl.catalog import BENCHMARKS

    class BudgetModel(VLLMModel):
        def setup_webserver(self):
            from nemo_gym.rollout_correlation import RolloutContextMiddleware
            app = super().setup_webserver()
            app.add_middleware(RolloutContextMiddleware)
            return app

        async def chat_completions(self, request, body):
            rollout = current_rollout_id()
            if rollout not in ledger.rows:
                raise HTTPException(400, 'Unknown evaluation rollout')
            async with ledger.locks[rollout]:
                row = ledger.rows[rollout]
                try:
                    if not row["infrastructure_error"] and ledger.total - row["output_tokens"] <= 0:
                        raise BudgetTerminal("output_budget")
                    allowance = ledger.allowance(rollout)
                    body = body.model_copy(deep=True)
                    prepared = self._preprocess_chat_completion_create_params(request, body.model_dump(exclude_unset=True))
                    tokenized = await self._resolve_client(request).create_tokenize(**self._get_tokenize_chat_body(prepared))
                    prompt_tokens = tokenized.get('count')
                    if prompt_tokens is None and isinstance(tokenized.get('tokens'), list):
                        prompt_tokens = len(tokenized['tokens'])
                    if type(prompt_tokens) is not int or prompt_tokens < 1:
                        raise ValueError('Backend tokenization did not return an input length')
                    allowance = min(allowance, ledger.context - prompt_tokens)
                    if allowance <= 0:
                        row['truncated'] = True
                        ledger.commit(rollout)
                        raise BudgetTerminal('context_window')
                    row['last_allowance'] = allowance
                    body.max_tokens = allowance
                    # The native converter may preserve a caller's newer spelling.
                    if hasattr(body, 'max_completion_tokens'):
                        body.max_completion_tokens = None
                    index = row['calls']
                    save(ledger.root / rollout / f'call-{index:04d}.request.json', body.model_dump(mode='json'))
                    response = await super().chat_completions(request, body)
                    raw = response.model_dump(mode='json')
                    save(ledger.root / rollout / f'call-{index:04d}.response.json', raw)
                    ledger.charge(rollout, raw)
                    return response
                except BudgetTerminal as terminal:
                    row["truncated"] = True
                    row["budget_terminal_reason"] = str(terminal)
                    ledger.commit(rollout)
                    raise HTTPException(400, {"type": "evaluation_budget_exhausted", "reason": str(terminal)}) from None
                except Exception:
                    row['infrastructure_error'] = True
                    ledger.commit(rollout)
                    raise

    class EvalResources(CodingResourcesServer):
        def model_post_init(self, context):
            self._sandbox = RemoteSandbox(self.config.sandbox_image,
                                           memory_mib=self.config.sandbox_memory_mib)
            self._service = None

        async def verify(self, request: Request, body: VerifyRequest) -> VerifyResponse:
            session = self._pop_session(request.session.get('mm_session', ''))
            if session is None:
                raise RuntimeError('Missing evaluation session')
            rollout = session.capture_rollout_id
            record = dict(sample_id=session.task.task_id, rollout_id=rollout,
                          route=session.route, official_scoring_run=False)
            target = ledger.root / rollout
            try:
                assert body.capture_rollout_id == rollout
                assert body.route == session.route and Task(**body.task) == session.task
                save(target / 'raw-gym-response.json', body.response.model_dump(mode='json'))
                budget = ledger.rows[rollout]
                failed = (body.response.status not in (None, 'completed') or body.response.error is not None
                          or (body.response.metadata or {}).get('failed') == 'true'
                          or (body.response.metadata or {}).get('interrupted') == 'true'
                          or budget['infrastructure_error']
                          or budget['calls'] < 1 or budget['output_tokens'] < 1)
                try:
                    artifact = BENCHMARKS[session.task.benchmark].artifact
                    source = await self._sandbox.read_artifact(session.handle, artifact)
                    (target / artifact).write_text(source)
                    record['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
                    record['artifact_present'] = True
                except ValueError:
                    record['artifact_present'] = False
                record.update(status='infrastructure_error' if budget['infrastructure_error'] else 'truncated' if budget.get('budget_terminal_reason') in ('output_budget', 'context_window') else 'infrastructure_error' if failed else 'truncated' if budget['truncated']
                              else 'completed' if record['artifact_present'] else 'candidate_output_failure',
                              budget=dict(budget))
                save(target / 'RESULT.json', record)
                # Placeholder transport reward is explicitly excluded from scoring/training.
                return VerifyResponse(**body.model_dump(), reward=0.0, mask_sample=True,
                                      evaluation_record=record, official_scoring_run=False)
            finally:
                if session.handle is not None:
                    await self._close(session.handle)
    return BudgetModel, EvalResources
