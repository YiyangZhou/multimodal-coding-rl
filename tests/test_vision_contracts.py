import io
import json
import tarfile

import pytest

from mmrl.project_bundle import parse_project, project_archive
from mmrl.vision_workflows import parse_workflows


def bundle(*files):
    return json.dumps({"files": [{"path": name, "content": content} for name, content in files]})


def test_full_stack_project_preserved_without_execution():
    files = parse_project(bundle(("start.sh", "python server.py"), ("server.py", "raise RuntimeError('never run on host')"),
                                 ("public/index.html", "<h1>App</h1>")))
    with tarfile.open(fileobj=io.BytesIO(project_archive(files)), mode="r:gz") as archive:
        assert {x.name for x in archive} == set(files)
        assert all(x.isfile() for x in archive)
        assert archive.extractfile("server.py").read() == files["server.py"]


@pytest.mark.parametrize("path", ["../outside", "/etc/passwd", "resources/logo.png", "prototypes/a.jpg",
                                  "workflow.json", "a/../../b", "a\\b", "a//b", "./a", "a\nname", "."])
def test_project_cannot_replace_inputs_or_escape_workspace(path):
    with pytest.raises(ValueError):
        parse_project(bundle(("start.sh", "true"), (path, "x")))


def test_duplicate_keys_paths_and_directory_collisions_rejected():
    with pytest.raises(ValueError, match="Duplicate JSON"):
        parse_project('{"files":[],"files":[]}')
    for files in [(('start.sh', 'true'), ('start.sh', 'false')),
                  (('start.sh', 'true'), ('app', 'x'), ('app/main.py', 'y'))]:
        with pytest.raises(ValueError):
            parse_project(bundle(*files))


def test_missing_start_script_and_oversized_project_rejected():
    with pytest.raises(ValueError, match="start.sh"):
        parse_project(bundle(("index.html", "<h1>hello</h1>")))
    with pytest.raises(ValueError, match="budget"):
        parse_project(bundle(("start.sh", "x" * 1024)), max_bytes=128)


def test_workflow_preserves_actions_and_interleaved_visual_states():
    source = [{"resolution": {"width": 375, "height": 812}, "content": [
        {"objective": "Submit", "actions": ["Enter an email", "Click Submit"], "validations": ["Confirmation shown"]},
        {"objective": "Return", "actions": ["Click Back"], "validations": []},
    ], "prototype": {"initial": {"idx": 0}, "submitted": {"idx": 1, "fullpage": False},
                      "final": {"idx": 8}}}]
    workflow = parse_workflows(source)[0]
    assert (workflow.width, workflow.height) == (375, 812)
    assert [(n.kind, n.index) for n in workflow.nodes] == [('visual', 0), ('functional', 0),
        ('visual', 1), ('functional', 1), ('visual', 8)]
    assert workflow.nodes[1].payload['actions'] == source[0]['content'][0]['actions']
    assert workflow.nodes[2].payload['fullpage'] is False


def test_same_prototype_in_distinct_workflows_is_not_deduplicated():
    result = parse_workflows([{'prototype': {'home': {'idx': 0}}}, {'prototype': {'home': {'idx': 0}}}])
    assert len(result) == 2 and result[0].index != result[1].index


@pytest.mark.parametrize('workflow', [[], [{}], [{'content': 'not-a-list'}],
    [{'prototype': {'../private': {}}}], [{'prototype': {'home': {'idx': True}}}],
    [{'prototype': {'home': {'fullpage': 'false'}}}],
    [{'content': [{'actions': [''], 'validations': []}]}]])
def test_invalid_workflow_is_not_silently_skipped(workflow):
    with pytest.raises(ValueError):
        parse_workflows(workflow)
