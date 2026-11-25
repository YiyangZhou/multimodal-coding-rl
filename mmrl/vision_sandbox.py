"""Deploy the entire policy project and verify its full workflow in OpenSandbox."""
import json
from pathlib import Path
import tempfile

from .project_bundle import parse_project, project_archive
from .render_transport import download_bounded
from .reward import InfrastructureError
from .vision_data import checked_file, verify_manifest
from .vision_reward import evaluate_workflows
from .vision_resources import write_resource_archive
from .vision_transport import VisionBrowserSession
from .vision_workflows import parse_workflows


def read_inputs(manifest):
    try:
        verify_manifest(manifest)
        workflows = parse_workflows(manifest["private_workflows"])
        root = Path(manifest["project_root"])
        references = {(item["workflow"], item["name"]): checked_file(root, item["path"]).read_bytes()
                      for item in manifest["private_visual_references"]}
        return workflows, references
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InfrastructureError("Vision2Web dataset is missing, changed or malformed") from exc


def deployment_ready(value, exit_code):
    if not isinstance(value, dict) or not isinstance(value.get("reason"), str):
        raise InfrastructureError("Invalid web deployment receipt")
    if value.get("status") == "ready" and exit_code == 0:
        return True
    if (value.get("status") == "execution_failed" and exit_code == 20
            and value["reason"] == "start_script_exited_without_application"
            and type(value.get("exit_code")) is int and value["exit_code"] >= 0):
        return False
    raise InfrastructureError("Web deployment did not produce a usable model measurement")


async def score_vision(sandbox, task, candidate, judge, audit=None):
    # This parser only packages text; generated scripts are never executed locally.
    try:
        files = parse_project(candidate)
    except ValueError as exc:
        return {"execution": 0.0, "reward": 0.0, "failure_kind": "mm_coding:invalid_artifact",
                "failure_reason": str(exc)}
    workflows, references = read_inputs(task.vision_manifest)
    async with sandbox.session() as handle:
        await sandbox.prepare(handle)
        browser = VisionBrowserSession(sandbox, handle)
        await browser.prepare()
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "project.tar.gz"
            archive.write_bytes(project_archive(files))
            await sandbox.provider.upload_file(handle, archive, "/workspace/project.tar.gz")
        unpack = await sandbox.provider.exec(handle,
            "tar -xzf /workspace/project.tar.gz -C /workspace", timeout_s=30)
        if unpack.error_type or unpack.return_code:
            raise InfrastructureError("Project transfer or extraction failed")
        assets = [r for r in task.vision_manifest["public_assets"] if r["path"].startswith("resources/")]
        root = Path(task.vision_manifest["project_root"])
        if assets:
            with tempfile.TemporaryDirectory() as tmp:
                archive = Path(tmp) / "resources.tar.gz"
                try:
                    write_resource_archive(root, assets, archive)
                except (OSError, ValueError, KeyError) as exc:
                    raise InfrastructureError("Resource archive preparation failed") from exc
                await sandbox.provider.upload_file(handle, archive, "/workspace/.mmrl/resources.tar.gz")
            unpack = await sandbox.provider.exec(handle,
                "tar -xzf /workspace/.mmrl/resources.tar.gz -C /workspace", timeout_s=120)
            if unpack.error_type or unpack.return_code:
                raise InfrastructureError("Resource archive transfer or extraction failed")
        # Private tests and reference prototypes are never staged with app code.
        await sandbox.upload(handle, "/workspace/.mmrl/resources.json", json.dumps(assets))
        await sandbox.upload(handle, "/workspace/.mmrl/deploy.py",
                             Path(__file__).with_name("vision_deploy.py").read_text())
        deployed = await sandbox.provider.exec(handle, "python /workspace/.mmrl/deploy.py", timeout_s=330)
        if deployed.error_type or deployed.return_code not in (0, 20, 30):
            raise InfrastructureError("Web deployment command or sandbox transport failed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deployment.json"
            await download_bounded(sandbox, handle, "/workspace/.mmrl/deployment.json", path, 32768)
            try:
                receipt = json.loads(path.read_text())
            except ValueError as exc:
                raise InfrastructureError("Unreadable web deployment receipt") from exc
        if audit is not None:
            audit.write("deployment.json", json.dumps(receipt))
        if not deployment_ready(receipt, deployed.return_code):
            return {"execution": 0.0, "reward": 0.0, "failure_kind": "mm_coding:execution_failed"}
        score, results = await evaluate_workflows(task.prompt, workflows, browser, judge, references)
        if not score["execution"]:
            score["failure_kind"] = "mm_coding:execution_failed"
        if audit is not None:
            for result in results:
                for index, visual in enumerate(result["visuals"]):
                    name = f"workflow-{result['workflow']}-visual-{index}.png"
                    audit.write(name, visual.pop("png"))
                    visual["file"] = name
            audit.write("workflows.json", json.dumps(results))
            audit.write("vision_reward.json", json.dumps(score))
        return score
