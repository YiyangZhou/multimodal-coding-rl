"""Real Quality judge workflow control against a synthetic SQLite-backed web app."""

import asyncio
import hashlib
import json
import os
import argparse
from pathlib import Path
import sys
import tempfile
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


async def main(args):
    from mmrl.sandbox import RemoteSandbox
    from mmrl.project_bundle import parse_project, project_archive
    from mmrl.vision_transport import VisionBrowserSession
    from mmrl.vision_judge import VisionJudgeController
    from mmrl.vision_workflows import parse_workflows
    from mmrl.vision_verifier import verify_workflow
    from mmrl.vision_reward import score_results

    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    image = args.image
    sandbox = RemoteSandbox(image, ttl=1800)

    class Judge(VisionJudgeController):
        calls = 0

        async def structured(self, *args, **kwargs):
            self.calls += 1
            return await super().structured(*args, **kwargs)

    judge = Judge(concurrency=1)
    report = {
        "passed": False,
        "image": image,
        "optimizer_updates": 0,
        "policy_model_calls": 0,
        "scope": "Synthetic browser/Quality judge integration; not official data or native training acceptance",
        "source_sha256": {
            name: hashlib.sha256((REPO / "mmrl" / name).read_bytes()).hexdigest()
            for name in [
                "vision_browser.py",
                "vision_transport.py",
                "vision_judge.py",
                "vision_verifier.py",
                "vision_workflows.py",
                "project_bundle.py",
                "vision_reward.py",
            ]
        },
    }
    try:
        async with sandbox.session() as handle:
            (out / "sandbox.json").write_text(
                json.dumps(
                    {
                        "descriptor": await sandbox.provider.serialize_handle(handle),
                        "deleted": False,
                    },
                    indent=2,
                )
            )
            await sandbox.prepare(handle)
            files = parse_project(
                (REPO / "tests/fixtures/vision_counter.json").read_text()
            )
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "project.tar.gz"
                archive.write_bytes(project_archive(files))
                await sandbox.provider.upload_file(
                    handle, archive, "/workspace/project.tar.gz"
                )
            result = await sandbox.provider.exec(
                handle,
                "mkdir -p /workspace/.mmrl && tar -xzf /workspace/project.tar.gz -C /workspace && nohup bash /workspace/start.sh > /workspace/.mmrl/deploy.log 2>&1 < /dev/null &",
                timeout_s=30,
            )
            if result.error_type or result.return_code:
                raise RuntimeError("Fixture deployment failed")
            browser = VisionBrowserSession(sandbox, handle)
            await browser.prepare()
            workflow = parse_workflows(
                [
                    {
                        "resolution": {"width": 1024, "height": 768},
                        "content": [
                            {
                                "objective": "Increment once",
                                "actions": ["Click the Increment button once."],
                                "validations": ["The page displays Count: 1."],
                            },
                            {
                                "objective": "Deliberately false assertion",
                                "actions": [],
                                "validations": ["The page displays Count: 99."],
                            },
                        ],
                        "prototype": {"after_increment": {"idx": 1, "fullpage": False}},
                    }
                ]
            )[0]
            value = await verify_workflow(workflow, browser, judge)
            if args.reference:
                score = await score_results(
                    "A counter webpage with an Increment button and persistent count.",
                    [workflow], [value], {(0, "after_increment"): args.reference.read_bytes()}, judge,
                )
                (out / "reward.json").write_text(json.dumps(score, indent=2) + "\n")
                report["reward_check"] = (score["functional_pass_rate"] == 0.5
                    and score["task_fidelity"] == 0.5 and len(score["visual_scores"]) == 1
                    and abs(score["reward"] - (0.2 + 0.4 * (0.5 + score["visual_fidelity"]))) < 1e-9)
            for i, item in enumerate(value["visuals"]):
                png = item.pop("png")
                name = f"visual-{i}.png"
                (out / name).write_bytes(png)
                item["file"] = name
                item["sha256"] = hashlib.sha256(png).hexdigest()
            (out / "workflow.json").write_text(json.dumps(value, indent=2))
            report["verdicts"] = [c["passed"] for c in value["cases"]]
            report["execution"] = value["execution"]
            report["visuals"] = len(value["visuals"])
            report["passed"] = (
                value["execution"] == 1
                and report["verdicts"] == [True, False]
                and report["visuals"] == 1
                and report.get("reward_check", True)
            )
        state = json.loads((out / "sandbox.json").read_text())
        state["deleted"] = True
        (out / "sandbox.json").write_text(json.dumps(state, indent=2))
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["frames"] = [
            {"file": x.filename, "line": x.lineno, "function": x.name}
            for x in traceback.extract_tb(exc.__traceback__)
        ]
    finally:
        report["judge_api_calls"] = judge.calls
        await sandbox.provider.aclose()
        (out / "RESULT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Published CPU renderer image pinned by digest"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        help="Independent counter fixture Count: 1 screenshot; enables real visual reward check")
    args = parser.parse_args()
    os.umask(0o077)
    # The Gym imports initialize Hydra; do not pass our CLI flags to it.
    sys.argv = [sys.argv[0]]
    raise SystemExit(asyncio.run(main(args)))
