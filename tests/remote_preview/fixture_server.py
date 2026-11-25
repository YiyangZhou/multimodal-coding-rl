"""Synthetic model replies, real Gym signed MCP dispatch; no training evidence."""
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import MagicMock

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer, BaseSeedSessionResponse, BaseSeedSessionRequest
from nemo_gym.server_utils import ServerClient, SESSION_ID_KEY
from nemo_gym.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.responses_streaming import flatten_namespace_tools, synthesize_responses_sse
from mmrl.process_tool import install_preview_route

ROOT=Path('/workspace/preview-check')
CONFIG=json.loads((ROOT/'fixture.json').read_text())
EVENTS=ROOT/'events.jsonl'
SESSIONS={}
AGENTS={}


def record(value):
    with EVENTS.open('a') as stream:
        stream.write(json.dumps(value)+'\n')


class FixturePreview:
    async def invoke(self, request_id, arguments):
        record({'event':'preview_called','request_id':str(request_id),'arguments':arguments})
        return {'executed':True,'preview_path':'/workspace/fixture-preview.png',
                'message':'Synthetic transport fixture; no score was computed.'}


class Resources(SimpleResourcesServer):
    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        sid=uuid4().hex
        request.session[SESSION_ID_KEY]=sid
        SESSIONS[sid]=FixturePreview()
        return BaseSeedSessionResponse()

    async def verify(self, body):
        raise NotImplementedError('This fixture must never score training data')

    def setup_webserver(self):
        app=super().setup_webserver()
        install_preview_route(app,SESSIONS.get)
        return app


resources=Resources(config=BaseResourcesServerConfig(host='127.0.0.1',port=18443,entrypoint='',name='mmrl',expose_tools_over_mcp=True),server_client=MagicMock(spec=ServerClient))
app=resources.setup_webserver()
maybe_auto_expose(resources,app)


@app.middleware('http')
async def require_fixture_auth(request, call_next):
    if request.url.path not in ('/health','/seed_session') and request.headers.get('authorization')!='Bearer '+CONFIG['token']:
        return JSONResponse({'error':'Unauthorized'},status_code=401)
    return await call_next(request)


@app.get('/health')
async def health():
    return {'fixture':True}


def preview_name(tools):
    for tool in tools:
        spec=tool.get('function',tool)
        if isinstance(spec,dict) and str(spec.get('name','')).endswith('preview'):
            return spec['name']
    return None


@app.post('/{agent}/v1/responses')
async def responses(agent: str, request: Request):
    body=await request.json()
    flat,namespaces=flatten_namespace_tools(body.get('tools',[]))
    name=preview_name(flat)
    items=body.get('input',[])
    has_result=isinstance(items,list) and any(isinstance(v,dict) and v.get('type')=='function_call_output' for v in items)
    record({'event':'model_request','agent':agent,'dialect':'responses','preview_tool':name,'has_tool_result':has_result})
    if not name:
        raise HTTPException(422,'CLI did not discover preview')
    if not has_result:
        output=[{'type':'function_call','id':'fc-fixture','call_id':'fixture-call','name':name,'arguments':'{}','status':'completed'}]
    else:
        output=[{'type':'message','id':'msg-fixture','role':'assistant','status':'completed','content':[{'type':'output_text','text':'Fixture complete.','annotations':[]}]}]
    result={'id':'resp-'+uuid4().hex,'object':'response','created_at':int(time.time()),'model':'policy','status':'completed',
        'output':output,'usage':{'input_tokens':20,'output_tokens':10,'total_tokens':30},'error':None,'incomplete_details':None}
    if body.get('stream'):
        return StreamingResponse(synthesize_responses_sse(result,namespaces),media_type='text/event-stream')
    return result


@app.post('/{agent}/v1/chat/completions')
async def chat(agent: str, request: Request):
    body=await request.json();name=preview_name(body.get('tools',[]))
    has_result=any(v.get('role')=='tool' for v in body.get('messages',[]))
    record({'event':'model_request','agent':agent,'dialect':'chat','preview_tool':name,'has_tool_result':has_result})
    if not name:raise HTTPException(422,'CLI did not discover preview')
    if not has_result:
        message={'role':'assistant','content':None,'tool_calls':[{'id':'fixture-call','type':'function','function':{'name':name,'arguments':'{}'}}]}
        finish='tool_calls'
    else:
        message={'role':'assistant','content':'Fixture complete.'};finish='stop'
    response={'id':'chatcmpl-'+uuid4().hex,'object':'chat.completion','created':int(time.time()),'model':body.get('model','policy'),
        'choices':[{'index':0,'message':message,'finish_reason':finish}],
        'usage':{'prompt_tokens':20,'completion_tokens':10,'total_tokens':30}}
    if not body.get('stream'):return response
    delta=dict(message)
    if 'tool_calls' in delta:delta['tool_calls']=[dict(v,index=i) for i,v in enumerate(delta['tool_calls'])]
    base={k:v for k,v in response.items() if k not in ('choices','usage')};base['object']='chat.completion.chunk'
    frames=[base|{'choices':[{'index':0,'delta':delta,'finish_reason':None}]},
            base|{'choices':[{'index':0,'delta':{},'finish_reason':finish}],'usage':response['usage']}]
    return StreamingResponse(iter(['data: '+json.dumps(v)+'\n\n' for v in frames]+['data: [DONE]\n\n']),media_type='text/event-stream')
