"""Fixed held-out examples against a caller-pinned generation weight version.

The controller must prevent refits during this call. These examples never enter
training queues. Rendering and quality use the same CodingService as Gym.
"""
import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from .audit import VerificationAudit
from .tasks import Task
from .reward import InfrastructureError


def chat_request(row, model, *, seed=42, max_tokens=32768):
    """Translate only policy-visible Gym input; never expose private task fields."""
    messages = []
    for message in row['responses_create_params']['input']:
        content = message['content']
        if isinstance(content, list):
            converted = []
            for part in content:
                if part['type'] == 'input_text':
                    converted.append({'type': 'text', 'text': part['text']})
                elif part['type'] == 'input_image':
                    converted.append({'type': 'image_url', 'image_url': {
                        'url': part['image_url'], 'detail': part.get('detail', 'auto')}})
                else:
                    raise ValueError('Unsupported fixed-case input part')
            content = converted
        messages.append({'role': message['role'], 'content': content})
    return {'model': model, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': 1.0, 'top_p': 1.0, 'seed': seed, 'stream': False,
            'chat_template_kwargs': {'enable_thinking': True}}


async def evaluate_cases(rows, endpoints, service, destination, *, weight_version,
                         seed=42, max_tokens=32768, concurrency=2, model_name=None):
    """Save complete responses/media, including failures, without training updates."""
    if not endpoints or len(rows) != 6 or len({r['task']['task_id'] for r in rows}) != 6:
        raise ValueError('Require six unique held-out cases and a generation endpoint')
    counts = {b: sum(r['task']['benchmark'] == b for r in rows)
              for b in ('manimbench_v1', '3dcodebench')}
    if counts != {'manimbench_v1': 3, '3dcodebench': 3}:
        raise ValueError('Require three animation and three 3D cases')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=3600, trust_env=False) as client:
        async def one(index, row):
            async with semaphore:
                case = destination / f'case-{index:02d}'
                case.mkdir()
                task = Task(**row['task'])
                record = {'task_id': task.task_id, 'benchmark': task.benchmark,
                          'weight_version': weight_version, 'seed': seed,
                          'max_tokens': max_tokens, 'prompt': task.prompt,
                          'status': 'incomplete'}
                try:
                    endpoint = endpoints[index % len(endpoints)].rstrip('/')
                    model = model_name
                    if model is None:
                        models = await client.get(endpoint + '/models')
                        models.raise_for_status()
                        model = models.json()['data'][0]['id']
                    body = chat_request(row, model, seed=seed, max_tokens=max_tokens)
                    record['request_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
                    response = await client.post(endpoint + '/chat/completions', json=body)
                    response.raise_for_status()
                    payload = response.json()
                    (case/'response.json').write_text(json.dumps(payload)+'\n')
                    choice = payload['choices'][0]
                    record['finish_reason'] = choice['finish_reason']
                    answer = choice['message'].get('content') or ''
                    (case/'answer.txt').write_text(answer)
                    if choice['finish_reason'] != 'stop' or not answer.strip():
                        record['status'] = 'truncated' if choice['finish_reason'] == 'length' else 'incomplete'
                    else:
                        audit = VerificationAudit(case/'verification', task=task, route='single',
                                                  response_id=payload['id'], status='completed',
                                                  capture_media=True)
                        score = await service.score(task, answer, audit=audit)
                        audit.finish(score)
                        record.update(status='scored', result=score, audit_path=str(audit.path))
                except (InfrastructureError, httpx.HTTPError, OSError, ValueError, KeyError, IndexError) as exc:
                    record.update(status='infrastructure_error', error_type=type(exc).__name__)
                    if isinstance(exc, httpx.HTTPStatusError):
                        record.update(http_status=exc.response.status_code,
                                      request_method=exc.request.method,
                                      request_path=exc.request.url.path)
                (case/'result.json').write_text(json.dumps(record,indent=2)+'\n')
                return record
        results = await asyncio.gather(*(one(i,r) for i,r in enumerate(rows)))
    report = {'weight_version': weight_version, 'seed': seed, 'cases': results,
              'scope': 'Fixed held-out examples, never training samples'}
    (destination/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
