"""Validate and order the pinned Vision2Web workflow contract.

Prototype idx=0 means initial state; idx=N means after functional case N-1.
The policy never receives these private verification instructions.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    kind: str
    index: int
    payload: dict


@dataclass(frozen=True)
class Workflow:
    index: int
    width: int
    height: int
    nodes: tuple[Node, ...]


def _strings(value, label):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"Invalid workflow {label}")
    return list(value)


def parse_workflows(value):
    if not isinstance(value, list) or not value or len(value) > 128:
        raise ValueError("Expected a nonempty bounded workflow list")
    workflows = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError("Invalid workflow object")
        resolution = raw.get("resolution", {})
        if not isinstance(resolution, dict):
            raise ValueError("Invalid workflow resolution")
        width, height = resolution.get("width", 1920), resolution.get("height", 1080)
        if any(type(n) is not int or not 100 <= n <= 4096 for n in (width, height)):
            raise ValueError("Invalid workflow viewport")
        content, prototypes = raw.get("content", []), raw.get("prototype", {})
        if not isinstance(content, list) or not isinstance(prototypes, dict):
            raise ValueError("Invalid functional or visual node collection")
        if len(content) > 128 or len(prototypes) > 128 or not (content or prototypes):
            raise ValueError("Workflow has no usable nodes or exceeds its budget")
        visual = {}
        for name, config in prototypes.items():
            if (not isinstance(name, str) or not name or "/" in name or "\\" in name
                    or name in {".", ".."} or any(ord(c) < 32 for c in name)
                    or not isinstance(config, dict)):
                raise ValueError("Invalid prototype identity")
            position, fullpage = config.get("idx", 0), config.get("fullpage", True)
            if type(position) is not int or position < 0 or position > 1024 or type(fullpage) is not bool:
                raise ValueError("Invalid visual checkpoint position")
            visual.setdefault(position, []).append(Node("visual", position, {"name": name, "fullpage": fullpage}))
        nodes = list(visual.get(0, []))
        for case_index, case in enumerate(content):
            if not isinstance(case, dict) or not isinstance(case.get("objective", ""), str):
                raise ValueError("Invalid functional case")
            nodes.append(Node("functional", case_index, {
                "objective": case.get("objective", ""),
                "actions": _strings(case.get("actions", []), "actions"),
                "validations": _strings(case.get("validations", []), "validations"),
            }))
            nodes.extend(visual.get(case_index + 1, []))
        for position in sorted(visual):
            if position > len(content):
                nodes.extend(visual[position])
        workflows.append(Workflow(index, width, height, tuple(nodes)))
    return workflows
