"""Real CPU OpenSandbox + Quality judge acceptance; no policy generation or trainer update."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

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


async def check(image, prepared, output):
    output.mkdir(parents=True, exist_ok=False)
    preparation = json.loads((prepared / "preparation.json").read_text())
    source = prepared / "train.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != preparation["files_sha256"][source.name]:
        raise ValueError("Prepared Manim records do not match their audit")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    selected = [next(row for row in rows if row["task_id"] == f"manimbench-v1/train/{index:04d}")
                for index in (0, 1, 139)]
    sandbox = RemoteSandbox(image, ttl=1200)
    report = {"image": image, "passed": False, "checks": [], "policy_calls": 0,
              "optimizer_updates": 0, "source_revision": preparation["revision"],
              "scope": "Three real reference-as-candidate measurements plus invalid-code boundary"}
    judge = None
    try:
        judge = CountingJudge(concurrency=1)
        service = CodingService(sandbox, judge)
        for row in selected:
            task = Task(**{key: value for key, value in row.items() if key != "split"})
            audit = VerificationAudit(output / "verification", task=task, route="single",
                                      response_id=task.task_id, status="completed")
            result = await service.score(task, "```python\n" + task.ground_truth + "\n```", audit=audit)
            audit.finish(result)
            passed = result.get("execution") == 1 and .2 <= result.get("reward", -1) <= 1
            item = {"task_id": task.task_id, "passed": passed, "result": result,
                    "audit": str(audit.path),
                    "reference_sha256": hashlib.sha256(task.ground_truth.encode()).hexdigest()}
            report["checks"].append(item)
            print(json.dumps(item), flush=True)
        before = judge.calls
        failed = await service.score(task, "```python\nThis is not valid Python !!!\n```")
        report["checks"].append({"case": "syntax_error_no_api",
            "passed": failed["execution"] == 0 and failed["reward"] == 0 and judge.calls == before})
        # A broken private reference is an environment/data failure, never a
        # valid candidate's zero reward. It must not reach the quality API.
        broken = Task("manimbench_v1", "broken-reference", task.prompt,
                      ground_truth="This is not valid Python !!!")
        try:
            await service.score(broken, task.ground_truth)
        except InfrastructureError:
            report["checks"].append({"case": "reference_failure_excluded", "passed": judge.calls == before})
        else:
            report["checks"].append({"case": "reference_failure_excluded", "passed": False})
        report["passed"] = all(row["passed"] for row in report["checks"])
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error_frames"] = [{"file": f.filename, "line": f.lineno, "function": f.name}
                                  for f in traceback.extract_tb(exc.__traceback__)]
    finally:
        report["judge_calls"] = judge.calls if judge is not None else 0
        await sandbox.provider.aclose()
        (output / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "result": str(output / "RESULT.json")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Published digest-pinned CPU image")
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Gym imports initialize Hydra. Only this script parses our command flags.
    sys.argv = [sys.argv[0]]
    raise SystemExit(asyncio.run(check(args.image, args.prepared, args.output)))
