"""Run the pinned real harnesses against synthetic model replies and native MCP."""
import json
import os
from pathlib import Path
import secrets
import signal
import shutil
import subprocess
import sys
import time
import urllib.request
import ssl

ROOT=Path('/workspace/preview-check')
sys.path[:0]=[str(ROOT/'gym_mount'),str(ROOT/'plugin')]
from mmrl.harness_tools import configure_preview_tools


def main():
    from PIL import Image
    Image.new('RGB',(32,32),'white').save('/workspace/fixture-preview.png')
    token=secrets.token_urlsafe(40)
    (ROOT/'fixture.json').write_text(json.dumps({'token':token}))
    from mmrl.tls import issue_certificate
    ca=ROOT/'ca.pem';ca_key=ROOT/'ca.key'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes',
        '-keyout',str(ca_key),'-out',str(ca),'-days','1','-subj','/CN=MMRL transport fixture CA',
        '-addext','basicConstraints=critical,CA:TRUE',
        '-addext','keyUsage=critical,keyCertSign,cRLSign'],
        check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    ca_key.chmod(0o600)
    cert,key=issue_certificate(ca,ca_key,'127.0.0.1',ROOT)
    subprocess.run(['openssl','verify','-x509_strict','-CAfile',str(ca),str(cert)],check=True)
    # Match RemoteSandbox.prepare: trust a public CA, serve a separate leaf.
    shutil.copyfile(ca,'/usr/local/share/ca-certificates/mmrl-fixture.crt')
    subprocess.run(['update-ca-certificates'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT/'gym_mount')+':'+str(ROOT/'plugin'),
        NODE_EXTRA_CA_CERTS=str(ca),SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt',REQUESTS_CA_BUNDLE='/etc/ssl/certs/ca-certificates.crt',
        OPENAI_API_KEY=token,HERMES_API_TIMEOUT='120',PYTHONDONTWRITEBYTECODE='1')
    os.environ.update({k:env[k] for k in ('PYTHONPATH','NODE_EXTRA_CA_CERTS','SSL_CERT_FILE','REQUESTS_CA_BUNDLE')})
    gateway='https://127.0.0.1:18443';ctx=ssl.create_default_context(cafile=str(ca))
    server_log=(ROOT/'server.log').open('w')
    server=subprocess.Popen(['/opt/gym/bin/python','-m','uvicorn','fixture_server:app','--host','127.0.0.1','--port','18443',
        '--ssl-certfile',str(cert),'--ssl-keyfile',str(key),'--no-access-log'],cwd=ROOT,env=env,stdout=server_log,stderr=subprocess.STDOUT)
    results=[]
    try:
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('Fixture server exited')
            try:
                with urllib.request.urlopen(gateway+'/health',context=ctx,timeout=2) as response:
                    if response.status==200:break
            except (OSError,urllib.error.URLError) as exc:
                (ROOT/'health-error.log').write_text(type(exc).__name__+': '+str(exc))
                time.sleep(.5)
        else:raise TimeoutError('Fixture server did not start')
        configs=json.loads((ROOT/'agent_configs.json').read_text())
        for agent in ('codex','opencode','hermes'):
            work=ROOT/agent;work.mkdir()
            (work/'gym_mount').symlink_to(ROOT/'gym_mount',target_is_directory=True)
            req=urllib.request.Request(gateway+'/seed_session',data=b'{}',headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req,context=ctx,timeout=10) as response:seed=json.load(response)
            seed.update(mmrl_preview_url=gateway+'/mcp',mmrl_preview_headers={'Authorization':'Bearer '+token})
            cfg=configs[agent]
            model_url=gateway+'/'+agent
            # Synthetic credentials only. Actual service/registry keys remain outside this sandbox.
            raw=json.dumps(cfg).replace('${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}',token).replace('__SANDBOX_MODEL_URL__',model_url)
            cfg=json.loads(raw)
            cfg['resources_server']={'type':'resources_servers','name':'fixture'}
            if agent=='codex':cfg.update(openai_base_url=model_url+'/v1',timeout=180,cwd='/workspace')
            elif agent=='opencode':cfg.update(timeout=180,workspace_root=str(work/'workspace'))
            else:cfg.update(model_server={'type':'responses_api_models','name':'policy'},max_turns=4,max_tokens=2048)
            cfg=configure_preview_tools(agent,cfg,seed,gateway=gateway)
            (work/'agent_config.json').write_text(json.dumps(cfg))
            (work/'model_url.txt').write_text(model_url)
            cls={'codex':'Codex','opencode':'OpenCode','hermes':'Hermes'}[agent]
            (work/'runner_config.json').write_text(json.dumps({'agent_module':f'responses_api_agents.{agent}_agent.app',
                'agent_class':cls+'Agent','agent_config_class':cls+'AgentConfig','cwd':'/workspace'}))
            (work/'request.json').write_text(json.dumps({'input':'Call the preview tool once, then finish. This is a synthetic tool transport test.'}))
            runner=(ROOT/'gym_mount/responses_api_agents/harness_agent/agent_runner.py').read_text()
            if agent=='codex':
                prefix = """import sys, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'gym_mount'))
import responses_api_agents.codex_agent.app as native_codex
original_diagnostics=native_codex.codex_exit_diagnostics
def fixture_diagnostics(stdout,stderr,returncode):
    value=stderr.decode(errors='replace')
    key=os.environ.get('OPENAI_API_KEY')
    if key:value=value.replace(key,'[SYNTHETIC_TOKEN]')
    (Path(__file__).resolve().parent/'cli.stderr').write_text(value)
    return original_diagnostics(stdout,stderr,returncode)
native_codex.codex_exit_diagnostics=fixture_diagnostics
"""
                runner=prefix+runner
            (work/'runner.py').write_text(runner)
            python='/opt/hermes/bin/python' if agent=='hermes' else '/opt/gym/bin/python'
            command=[python,'-m','mmrl.hermes_runner',str(work/'runner.py')] if agent=='hermes' else [python,str(work/'runner.py')]
            before=(ROOT/'events.jsonl').read_text().splitlines() if (ROOT/'events.jsonl').exists() else []
            with (work/'run.log').open('w') as stream:
                try:
                    process=subprocess.Popen(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    code=process.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait();code=124
            events=[json.loads(v) for v in (ROOT/'events.jsonl').read_text().splitlines()[len(before):]] if (ROOT/'events.jsonl').exists() else []
            data=json.loads((work/'response.json').read_text()) if (work/'response.json').exists() else {}
            calls=[v for v in events if v['event']=='preview_called']
            model=[v for v in events if v['event']=='model_request' and v['agent']==agent]
            messages=[v for v in data.get('output',[]) if v.get('type')=='message' and v.get('role')=='assistant']
            final_text=''.join(v.get('text','') for m in messages for v in m.get('content',[]) if v.get('type')=='output_text')
            # Native Codex omits the optional top-level status after checking its
            # CLI terminal turn. Require the actual final answer, never exit alone.
            completed=(data.get('status') in (None,'completed') and not data.get('error')
                       and not data.get('incomplete_details') and final_text=='Fixture complete.'
                       and messages and all(m.get('status')=='completed' for m in messages))
            passed=bool(code==0 and completed and bool(calls) and any(v['has_tool_result'] for v in model))
            results.append({'agent':agent,'passed':passed,'exit_code':code,'response_status':data.get('status'),'final_answer_verified':bool(completed),
                'preview_calls':len(calls),'model_requests':len(model),'received_tool_feedback':any(v['has_tool_result'] for v in model),
                'preview_names':sorted({v['preview_tool'] for v in model if v['preview_tool']})})
            (ROOT/'result.json').write_text(json.dumps({'passed':len(results)==3 and all(r['passed'] for r in results),'results':results,
                'synthetic_model':True,'real_cli':True,'native_signed_gym_mcp':True,'model_inference':False,'reward_scoring':False,'learner_update':False},indent=2)+'\n')
    finally:
        server.terminate()
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:server.kill();server.wait()
        server_log.close()
    print((ROOT/'result.json').read_text())
    return all(r['passed'] for r in results)

if __name__=='__main__':raise SystemExit(0 if main() else 1)
