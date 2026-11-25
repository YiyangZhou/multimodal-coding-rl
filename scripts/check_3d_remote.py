"""Canonical published-image 3D execution/Quality judge calibration, not policy evaluation."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

from download_benchmarks import verified
from mmrl.audit import VerificationAudit
from mmrl.reward import QualityJudge, InfrastructureError
from mmrl.sandbox import RemoteSandbox
from mmrl.service import CodingService
from mmrl.tasks import Task

class CountingJudge(QualityJudge):
    calls = 0
    async def grade(self, *args, **kwargs):
        self.calls += 1
        return await super().grade(*args, **kwargs)

async def check(image, data, output):
    lock = json.loads((Path(__file__).resolve().parents[1] /
                       "benchmarks/sources.lock.json").read_text())["datasets"]["3dcodebench"]
    artifacts = [a for a in lock["artifacts"] if "Beetle_seed0/" in a["path"]]
    if len(artifacts) != 3 or not all(verified(data / a["destination"], a) for a in artifacts):
        raise ValueError("Official fixture integrity failed")
    output.mkdir(parents=True, exist_ok=False)
    base = data / "3dcodebench/3DCodeBench/Beetle_seed0"
    task = Task('3dcodebench', 'Beetle_seed0', (base / 'prompt_instruction.txt').read_text(),
                ground_truth=(base / 'Beetle_seed0.py').read_text())
    sandbox = RemoteSandbox(image, ttl=1800)
    judge = CountingJudge(concurrency=1)
    report = {'passed': False, 'checks': [], 'policy_calls': 0, 'optimizer_updates': 0,
              'image': image, 'source_revision': lock['revision'],
              'scope': 'Official reference-as-candidate; canonical four views, 512 pixels, 64 samples; not Super performance'}
    try:
        service = CodingService(sandbox, judge)
        audit = VerificationAudit(output / 'verification', task=task, route='single',
                                  response_id=task.task_id, status='completed')
        result = await service.score(task, task.ground_truth, audit=audit)
        audit.finish(result)
        report['checks'].append({'case': 'official_reference', 'result': result,
                                'passed': result['execution'] == 1 and .2 <= result['reward'] <= 1,
                                'audit': str(audit.path)})
        print(json.dumps(report['checks'][-1]), flush=True)
        before = judge.calls
        result = await service.score(task, 'This is invalid Python !!!')
        report['checks'].append({'case': 'invalid_candidate_no_api',
                                'passed': result['execution'] == 0 and result['reward'] == 0 and judge.calls == before})
        broken = Task('3dcodebench', 'broken-reference', task.prompt, ground_truth='invalid Python !!!')
        try:
            await service.score(broken, task.ground_truth)
        except InfrastructureError:
            report['checks'].append({'case': 'invalid_reference_excluded', 'passed': judge.calls == before})
        else:
            report['checks'].append({'case': 'invalid_reference_excluded', 'passed': False})
        report['passed'] = all(item['passed'] for item in report['checks'])
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        report['error_frames'] = [{'file': f.filename, 'line': f.lineno, 'function': f.name}
                                  for f in traceback.extract_tb(exc.__traceback__)]
    finally:
        report['judge_calls'] = judge.calls
        await sandbox.provider.aclose()
        (output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': report['passed'], 'result': str(output / 'RESULT.json')}), flush=True)
    return 0 if report['passed'] else 1

if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    raise SystemExit(asyncio.run(check(args.image, args.data, args.output)))
