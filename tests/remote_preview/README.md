# Harness preview interoperability

The pinned OpenCode, Codex and Hermes CLIs call the real signed Gym MCP preview
route inside the remote sandbox image. Model replies and rendered content are
synthetic. This verifies discovery, authentication, tool feedback and final
completion; it does not verify policy inference, capture lineage, rewards or
training updates.

Provision an ephemeral sandbox using the image in `verified-result.json`, then
stage this directory at `/workspace/preview-check` with:

- `gym_mount/`: the locked Gym snapshot with capture and process-tools patches.
- `plugin/mmrl/`: this repository's Python package.
- `agent_configs.json`: a mapping from `codex`, `opencode`, and `hermes` to the
  corresponding `agent_kwargs` from a composed `gym.json`. Keep the generated
  environment placeholders; do not resolve real API credentials into this file.

Run `/opt/gym/bin/python /workspace/preview-check/run_remote.py` with working
directory `/workspace/preview-check`. Preserve `result.json`, `events.jsonl`,
`server.log` and each agent's `response.json` before deleting the sandbox.
The fixture creates its own temporary credentials and CA, uses the production
leaf-certificate issuer, and leaves certificate verification enabled. No judge
or production model API is called.

All three CLIs must exit successfully, receive tool feedback, and return the
exact completed fixture answer without an error or incomplete response. Native
Codex omits the optional top-level response status after checking CLI terminal
completion; the test therefore also checks completed assistant messages.

The recorded result passed all three CLIs in one sandbox, which was deleted
after collecting evidence.
