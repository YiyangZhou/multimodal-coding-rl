"""Real CodingService/OpenSandbox/Quality judge checks using a synthetic full-stack task."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


async def main(args):
    from PIL import Image
    from mmrl.audit import VerificationAudit
    from mmrl.reward import InfrastructureError
    from mmrl.sandbox import RemoteSandbox
    from mmrl.service import CodingService
    from mmrl.tasks import Task, gym_row
    from mmrl.vision_data import prepare_project
    from mmrl.vision_judge import VisionJudgeController

    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    project = out / "fixture"
    (project / "prototypes").mkdir(parents=True)
    (project / "resources").mkdir()
    (project / "resources/asset.txt").write_text("Synthetic fixture resource\n")
    with Image.open(args.reference) as image:
        image.convert("RGB").save(project / "prototypes/after_increment.jpg", quality=95)
    (project / "prd.md").write_text("Implement a database-backed counter with an Increment button.")
    (project / "workflow.json").write_text(json.dumps([{
        "resolution": {"width": 1024, "height": 768},
        "content": [
            {"objective": "Increment once", "actions": ["Click the Increment button once."],
             "validations": ["The page displays Count: 1."]},
            {"objective": "Deliberately false assertion", "actions": [],
             "validations": ["The page displays Count: 99."]},
        ], "prototype": {"after_increment": {"idx": 1, "fullpage": False}},
    }]))
    manifest = prepare_project(project, "website")
    task = Task("vision2web", manifest["task_id"], manifest["requirements"],
                [str(project / "prototypes/after_increment.jpg")],
                category="website", vision_manifest=manifest)
    (out / "gym-row.json").write_text(json.dumps(gym_row(task, "single")))
    sandbox = RemoteSandbox(args.image, ttl=1800)

    class Judge(VisionJudgeController):
        calls = 0

        async def structured(self, *positional, **kwargs):
            self.calls += 1
            return await super().structured(*positional, **kwargs)

    judge = Judge(concurrency=1)
    service = CodingService(sandbox, judge)
    report = {"passed": False, "image": args.image, "policy_model_calls": 0, "optimizer_updates": 0,
              "scope": "Synthetic full CodingService path; not official dataset or native Super RL admission",
              "source_sha256": {name: hashlib.sha256((REPO / "mmrl" / name).read_bytes()).hexdigest()
                                for name in ("vision_deploy.py", "vision_sandbox.py", "vision_reward.py",
                                             "vision_data.py", "vision_judge.py", "tasks.py", "service.py", "audit.py")},
              "cases": []}
    candidates = [
        ("valid", (REPO / "tests/fixtures/vision_counter.json").read_text()),
        ("app_failure", json.dumps({"files": [{"path": "start.sh", "content": "exit 1\n"}]})),
        ("network_failure", json.dumps({"files": [{"path": "start.sh", "content": "echo EAI_AGAIN >&2\nexit 1\n"}]})),
    ]
    try:
        for name, answer in candidates:
            before = judge.calls
            audit = VerificationAudit(out / "audit", task=task, route="single", response_id=name, status="completed")
            try:
                score = await service.score(task, "```json\n" + answer + "\n```", audit=audit)
                audit.finish(score)
                passed = (name == "valid" and score["execution"] == 1
                          and score["functional_pass_rate"] == 0.5
                          and score["functional_cases"] == 2
                          and len(score["visual_scores"]) == 1
                          and abs(score["reward"] - (0.2 + 0.4 * (0.5 + score["visual_fidelity"]))) < 1e-9
                          or name == "app_failure" and score["execution"] == score["reward"] == 0
                          and judge.calls == before)
                report["cases"].append({"name": name, "passed": passed, "score": score,
                                        "api_calls": judge.calls - before, "audit": str(audit.path.relative_to(out))})
            except InfrastructureError:
                if name != "network_failure":
                    raise
                report["cases"].append({"name": name, "passed": judge.calls == before,
                                        "excluded": True, "api_calls": judge.calls - before})
        report["passed"] = all(case["passed"] for case in report["cases"]) and len(report["cases"]) == 3
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["frames"] = [{"file": f.filename, "line": f.lineno, "function": f.name}
                            for f in traceback.extract_tb(exc.__traceback__)]
    finally:
        await sandbox.provider.aclose()
        report["judge_api_calls"] = judge.calls
        (out / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference", type=Path, required=True,
                        help="Independent Count: 1 reference screenshot from check_vision_remote.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    sys.argv = [sys.argv[0]]
    raise SystemExit(asyncio.run(main(args)))
