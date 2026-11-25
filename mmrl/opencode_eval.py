"""Opt-in OpenCode evaluation archive instrumentation; original source stays intact.

Pinned to the integrated Gym source and OpenCode 1.17.11 prompt semantics.
Call install(stdin_prompt=True) in a new evaluation worker, before provisioning.
"""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import inspect
from . import opencode_exit_diagnostics as diag

TARGET = 'responses_api_agents/opencode_agent/app.py'
ORIGINAL_SHA = '70c010e93e0d442a010dc35a2d19b81623c8cca615e7de06f96cabe7da8c6b37'
STDIN_SHA = '9691a7b5dbe7300cb792b14ac154604c078cf77a08292fd170e4d44517ba1719'

def patch_source(source, *, stdin_prompt=False):
    source_sha = hashlib.sha256(source).hexdigest()
    if source_sha not in (ORIGINAL_SHA, STDIN_SHA):
        raise ValueError('unrecognized OpenCode source')
    text = source.decode()
    if stdin_prompt and source_sha == ORIGINAL_SHA:
        changes = [
            ('        cmd.append(prompt)', '        piped_prompt = (chr(34) + prompt.replace(chr(34), chr(92) + chr(34)) + chr(34)) if " " in prompt else prompt'),
            ('                stdout=asyncio.subprocess.PIPE,', '                stdin=asyncio.subprocess.PIPE,\n                stdout=asyncio.subprocess.PIPE,'),
            ('proc.communicate(), timeout=self.config.timeout', 'proc.communicate(piped_prompt.encode("utf-8")), timeout=self.config.timeout'),
        ]
        for old, new in changes:
            if text.count(old) != 1:
                raise ValueError('stdin source anchor mismatch')
            text = text.replace(old, new)
    replacements = [
        ('        try:\n            timed_out = False\n            proc = await asyncio.create_subprocess_exec(',
         '        diagnostic_path = Path.cwd() / FILENAME\n        proc = None\n        stderr = b""\n        timed_out = False\n        try:\n            timed_out = False\n            proc = await asyncio.create_subprocess_exec('),
        ('        finally:\n            shutil.rmtree(work_dir, ignore_errors=True)',
         '        finally:\n            save(diagnostic_path, data_home / "opencode" / "opencode.db",\n                 getattr(proc, "returncode", None), timed_out, stderr)\n            shutil.rmtree(work_dir, ignore_errors=True)'),
        ('LOG.warning("opencode exited %d: %s", proc.returncode, stderr.decode(errors="replace")[:500])',
         'LOG.warning("opencode exited %d", proc.returncode)'),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('source anchor mismatch')
        text = text.replace(old, new)
    # Embed the tiny stdlib helper: source archive only contains agent_pkg+nemo_gym.
    marker = 'if __name__ == "__main__":'
    if text.count(marker) != 1:
        raise ValueError('main anchor mismatch')
    return text.replace(marker, inspect.getsource(diag) + '\n\n' + marker).encode()

def rewrite_archive(path, *, stdin_prompt=False):
    path = Path(path)
    temp = path.with_name(path.name + '.exit-diagnostics.tmp')
    count = 0
    try:
        with tarfile.open(path, 'r:gz') as old, tarfile.open(temp, 'w:gz') as new:
            for member in old:
                data = old.extractfile(member).read() if member.isfile() else None
                if member.name == TARGET:
                    data = patch_source(data, stdin_prompt=stdin_prompt)
                    member.size = len(data)
                    count += 1
                new.addfile(member, io.BytesIO(data) if data is not None else None)
        if count != 1:
            raise ValueError('expected one agent source')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return path

def install(*, stdin_prompt=False):
    from responses_api_agents.harness_agent.app import HarnessAgent
    previous_build = HarnessAgent._build_gym_tar
    previous_download = HarnessAgent._download_json
    def build(self):
        if self.config.agent == 'opencode' and stdin_prompt:
            if self.config.agent_kwargs.get('opencode_version') != '1.17.11':
                raise ValueError('stdin prompt normalization requires OpenCode 1.17.11')
        archive = previous_build(self)
        return rewrite_archive(archive, stdin_prompt=stdin_prompt) if self.config.agent == 'opencode' else archive
    async def download(self, handle, path):
        response = await previous_download(self, handle, path)
        if self.config.agent != 'opencode' or path != self._box_path(handle, '/work/response.json'):
            return response
        # Same sandbox handle + directory binds the diagnostic to this response/case.
        try:
            value = await previous_download(self, handle, self._box_path(handle, '/work/' + diag.FILENAME))
            value = diag.validate(value)
            encoded = json.dumps(value, separators=(',', ':')) if value is not None else None
        except Exception:
            encoded = None
        metadata = dict(response.get('metadata') or {})
        metadata['opencode_exit_diagnostics'] = encoded or '{"schema_version":1,"collection":"unavailable"}'
        return response | {'metadata': metadata}
    HarnessAgent._build_gym_tar = build
    HarnessAgent._download_json = download
