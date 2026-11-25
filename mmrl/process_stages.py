"""Trusted intermediate artifact collection, independent verification and audit.

The controller supplies capture identities; these are not agent request fields.
The opt-in process resource binds this collector to signed preview sessions.
"""
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .audit import VerificationAudit
from .catalog import BENCHMARKS
from .process_capture import CapturedStageScore
from .reward import InfrastructureError


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _json_hash(value):
    return _hash(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def _save(path, record):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


class StageCollector:
    def __init__(self, *, rollout_id, task, route, sandbox, service,
                 render_conditions, audit_directory):
        if not rollout_id or not isinstance(render_conditions, dict) or not render_conditions:
            raise ValueError('Explicit rollout identity and pinned rendering/scoring conditions required')
        self.rollout_id, self.task, self.route = rollout_id, task, route
        self.sandbox, self.service = sandbox, service
        self.directory = Path(audit_directory)
        self._task = asdict(task)
        self._conditions = json.loads(json.dumps(render_conditions, allow_nan=False))
        self._reference_hashes = self._references()
        self.conditions_signature = _json_hash({
            'task': self._task, 'conditions': self._conditions,
            'reference_sha256': self._reference_hashes,
        })
        self._cache, self._events, self._pending = {}, {}, {}
        self._previews = {}
        self._lock = asyncio.Lock()

    def _references(self):
        try:
            return {name: _hash(Path(name).read_bytes()) for name in self.task.references}
        except OSError as exc:
            raise InfrastructureError('Stage reference evidence is unavailable') from exc

    def _check_context(self):
        if asdict(self.task) != self._task or self._references() != self._reference_hashes:
            raise InfrastructureError('Task evidence changed during process scoring')

    async def capture_from_sandbox(self, handle, *, model_call_id, render_id):
        """Snapshot the catalog artifact; never accept a model-provided score or PNG.

        Idempotent render IDs must come from the trusted tool/capture observer.
        The caller must join them to the selected native terminal chain later.
        """
        if not model_call_id or not render_id:
            raise ValueError('Trusted model-call and render-event identities required')
        async with self._lock:
            self._check_context()
            if render_id in self._events:
                existing = self._events[render_id]
                if existing.model_call_id != model_call_id:
                    raise InfrastructureError('Render identity was reused for another model call')
                # A late retry refers to the sealed snapshot, even if the agent
                # has subsequently modified its live workspace.
                return existing
            try:
                source = await self.sandbox.read_artifact(handle, BENCHMARKS[self.task.benchmark].artifact)
            except ValueError:
                source = ''  # Actual missing/invalid artifact, not a transport failure.
            except OSError as exc:
                raise InfrastructureError('Intermediate artifact transfer failed') from exc
            source_signature = _hash(source.encode())
            identity = model_call_id, source_signature
            if self._pending.setdefault(render_id, identity) != identity:
                raise InfrastructureError('Unfinished render identity was reused for different evidence')
            audit = VerificationAudit(self.directory, task=self.task, route=self.route,
                                      response_id='process-stage', status='completed',
                                      rollout_id=self.rollout_id)
            record = {'complete': False, 'rollout_id': self.rollout_id,
                      'model_call_id': model_call_id, 'render_id': render_id,
                      'source_signature': source_signature,
                      'render_conditions_signature': self.conditions_signature}
            try:
                _save(audit.path / 'stage.json', record)
                cache_key = source_signature, self.conditions_signature
                if cache_key in self._cache:
                    score, original_audit = self._cache[cache_key]
                    audit.write('answer.txt', source)
                    record['cached_verification'] = str(original_audit)
                else:
                    score = await self.service.score(self.task, source, harness_artifact=True, audit=audit)
                    original_audit = audit.path
                self._check_context()
                if score.get('mask_sample') or score.get('failure_kind') == 'mm_coding:infrastructure':
                    raise InfrastructureError('Stage did not produce a valid quality measurement')
                reward = score['reward']
                if (type(reward) not in (int, float) or not 0 <= reward <= 1
                        or score.get('execution') not in (0, 1)):
                    raise InfrastructureError('Stage verifier returned an invalid result')
                preview = original_audit / 'render.png' if score['execution'] == 1 else None
                preview_hash = _hash(preview.read_bytes()) if preview else None
                audit.finish(score)
                stage = CapturedStageScore(self.rollout_id, model_call_id, render_id,
                                           float(reward), source_signature, self.conditions_signature)
                _save(audit.path / 'stage.json', {**record, 'complete': True, 'stage': asdict(stage)})
                self._cache.setdefault(cache_key, (dict(score), audit.path))
                self._events[render_id] = stage
                self._previews[render_id] = (preview, preview_hash)
                return stage
            except OSError as exc:
                raise InfrastructureError('Intermediate verification evidence could not be saved') from exc

    @property
    def stages(self):
        return tuple(self._events.values())

    def preview_file(self, render_id):
        """Trusted image path for tool feedback; never return judge scores/reasons."""
        if render_id not in self._events or render_id not in self._previews:
            raise InfrastructureError('Preview event is not sealed')
        path, expected = self._previews[render_id]
        if path is not None:
            try:
                if _hash(path.read_bytes()) != expected:
                    raise InfrastructureError('Preview image changed after verification')
            except OSError as exc:
                raise InfrastructureError('Verified preview image is unavailable') from exc
        return path
