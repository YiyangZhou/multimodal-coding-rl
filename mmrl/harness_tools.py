"""Translate trusted Gym preview metadata into the three pinned CLI configs."""
from copy import deepcopy
from urllib.parse import urlsplit


SESSION_HEADER = "X-NeMo-Gym-Session-Token"


def configure_preview_tools(agent, agent_config, seed, *, gateway):
    result = deepcopy(agent_config)
    if not seed or seed.get("mmrl_preview_url") is None:
        return result
    if agent not in {"codex", "opencode", "hermes"}:
        raise ValueError("This harness has no pinned preview-tool integration")
    url = seed["mmrl_preview_url"]
    supplied, allowed = urlsplit(url), urlsplit(str(gateway))
    if (supplied.scheme != "https" or supplied.path != "/mcp" or supplied.query
            or supplied.fragment or supplied.username or supplied.password
            or allowed.scheme != "https" or not allowed.hostname
            or allowed.username or allowed.password or allowed.query or allowed.fragment
            or allowed.path not in {"", "/", "/v1", "/v1/"}
            or (supplied.hostname, supplied.port or 443) != (allowed.hostname, allowed.port or 443)):
        raise ValueError("Preview service must use the configured private capture gateway")
    metadata = seed.get("mcp") or {}
    headers = {**(seed.get("mmrl_preview_headers") or {}), **(metadata.get("headers") or {})}
    if (metadata.get("transport") != "http"
            or set(headers) != {"Authorization", SESSION_HEADER}
            or any(not isinstance(v, str) or not v or "\n" in v or "\r" in v for v in headers.values())
            or not headers["Authorization"].startswith("Bearer ")):
        raise ValueError("Preview access requires gateway and signed Gym-session credentials")
    if agent == "codex":
        result.setdefault("extra_config", {}).setdefault("mcp_servers", {})["mmrl"] = {
            "url": url, "http_headers": headers, "enabled": True, "required": True,
            "enabled_tools": ["preview"], "startup_timeout_sec": 60, "tool_timeout_sec": 600,
            "supports_parallel_tool_calls": False,
        }
    elif agent == "opencode":
        config = result.setdefault("opencode_config", {})
        config.setdefault("mcp", {})["mmrl"] = {
            "type": "remote", "url": url, "headers": headers, "enabled": True,
            "oauth": False, "timeout": 600000,
        }
        permissions = config.setdefault("permission", {})
        if not isinstance(permissions, dict):
            raise ValueError("Preview integration requires explicit OpenCode tool permissions")
        permissions["mmrl_*"] = "allow"
    else:
        result.setdefault("mcp_servers", {})["mmrl"] = {
            "url": url, "headers": headers, "timeout": 600, "connect_timeout": 60,
            "tools": {"include": ["preview"]}, "sampling": {"enabled": False},
        }
        enabled = result.get("enabled_toolsets")
        if enabled is not None and "mcp-mmrl" not in enabled:
            result["enabled_toolsets"] = [*enabled, "mcp-mmrl"]
        required = result.setdefault("required_mcp_tools", [])
        if "mcp_mmrl_preview" not in required:
            required.append("mcp_mmrl_preview")
    return result
