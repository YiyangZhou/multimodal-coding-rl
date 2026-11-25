import json
from types import SimpleNamespace
import pytest
from fastapi import FastAPI, Request
import httpx
from history_ledger_fixture import USER_1,ASSISTANT_1,USER_2,_record_call_1
from nemo_gym.token_id_capture.lineage import FileLineageStore,InMemoryLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext,CapturePoisonedError,set_token_sink,reset_token_sink
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.base_responses_api_model import SimpleResponsesAPIModel,install_model_call_capture,ModelCallCaptureConfig

class BackendReached(RuntimeError):pass
class Stub:
    calls=0
    async def responses(self,body):self.calls+=1;raise BackendReached()
    async def chat_completions(self,body):self.calls+=1;raise BackendReached()
    _invoke_responses=SimpleResponsesAPIModel._invoke_responses
    _invoke_chat_completions=SimpleResponsesAPIModel._invoke_chat_completions
    _stream_responses=SimpleResponsesAPIModel._stream_responses
    responses_dispatch=SimpleResponsesAPIModel.responses_dispatch
    chat_completions_dispatch=SimpleResponsesAPIModel.chat_completions_dispatch

@pytest.fixture(params=['file','memory'])
def store(request,tmp_path):return FileLineageStore(tmp_path) if request.param=='file' else InMemoryLineageStore()

async def invoke(store,items,*,enabled=True,cid='c2',dialect='responses',external=True):
    context=CaptureContext('r1',cid,None,lineage_store=store,external_staging=external,fail_fast_poisoned_rollouts=enabled)
    token=set_token_sink(context);stub=Stub()
    try:
        body=SimpleNamespace(input=items,messages=items)
        method=stub._invoke_responses if dialect=='responses' else stub._invoke_chat_completions
        await method(None,body)
    finally:
        reset_token_sink(token)
    return stub

@pytest.mark.parametrize('dialect',['responses','chat'])
async def test_first_unresolved_and_all_later_calls_block_backend(store,dialect):
    await _record_call_1(store)
    # A matching assistant fingerprint with rewritten request prefix is unresolved.
    with pytest.raises(CapturePoisonedError):
        await invoke(store,[{'role':'user','content':'rewritten'},ASSISTANT_1,USER_2],dialect=dialect)
    before=await store.manifest('r1');assert len(before['failures'])==1
    for cid,items in [('retry',[USER_1,ASSISTANT_1,USER_2]),('new-root',[USER_1])]:
        with pytest.raises(CapturePoisonedError):await invoke(store,items,cid=cid,dialect=dialect)
    assert await store.manifest('r1')==before

async def test_default_preserves_public_unresolved_rejection(store):
    from nemo_gym.base_responses_api_model import RolloutCaptureFailed
    await _record_call_1(store)
    with pytest.raises(RolloutCaptureFailed):
        await invoke(store,[{'role':'assistant','content':'unknown'}],enabled=False)
    assert (await store.manifest('r1'))['failures']
    with pytest.raises(RolloutCaptureFailed):
        await invoke(store,[USER_1],enabled=False,cid='root')

async def test_gate_preserves_clean_root_and_continuation(store):
    with pytest.raises(BackendReached):await invoke(store,[USER_1])
    await _record_call_1(store)
    with pytest.raises(BackendReached):await invoke(store,[USER_1,ASSISTANT_1,USER_2])

async def test_gate_ignores_non_external_capture(store):
    await store.record_failure('r1','failed','unresolved_parent')
    with pytest.raises(BackendReached):await invoke(store,[USER_1],external=False)

async def test_gate_fails_closed_on_manifest_read_error(store,monkeypatch):
    async def broken(*args):raise ValueError('manifest unavailable')
    monkeypatch.setattr(store,'manifest',broken)
    with pytest.raises(ValueError,match='manifest unavailable'):await invoke(store,[USER_1])

@pytest.mark.parametrize('dialect,stream',[('responses',False),('responses',True),('chat/completions',False),('chat/completions',True)])
async def test_real_dispatch_terminal_wire_and_zero_backend_calls(tmp_path,monkeypatch,dialect,stream):
    store=FileLineageStore(tmp_path/'ledger');await _record_call_1(store)
    import nemo_gym.base_responses_api_model as api
    monkeypatch.setattr(api,'installed_lineage_store',lambda:store)
    monkeypatch.setenv('NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN','test-only')
    app=FastAPI();stub=Stub()
    @app.post('/v1/responses')
    async def responses(request:Request):return await stub.responses_dispatch(request,await request.json())
    @app.post('/v1/chat/completions')
    async def chat(request:Request):return await stub.chat_completions_dispatch(request,await request.json())
    install_model_call_capture(app,ModelCallCaptureConfig(),global_config_dict={'token_id_capture':{'enabled':True,'external_staging':True,'rebuild_response':False,'dir':str(tmp_path/'tokens'),'fail_fast_poisoned_rollouts':True}})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        for items in [[{'role':'user','content':'rewritten'},ASSISTANT_1,USER_2],[USER_1]]:
            body={'model':'policy','stream':stream,('input' if dialect=='responses' else 'messages'):items}
            r=await client.post('/ng-rollout/r1/training-token-capture/v1/'+dialect,json=body)
            if dialect=='responses' and stream:
                assert r.status_code==200
                events=[json.loads(l[6:]) for l in r.text.splitlines() if l.startswith('data: ') and l[6:]!='[DONE]']
                terminal=next(x for x in events if x['type']=='response.failed')['response']
                assert terminal['status']=='failed' and terminal['output']==[]
                assert terminal['error']['code']=='capture_poisoned'
            else:
                assert r.status_code==400,r.text
                assert r.json()['error']['type']=='infrastructure_error'
                assert r.json()['error']['code']=='capture_poisoned'
    assert stub.calls==0
    assert (await store.manifest('r1'))['failures']

def test_setting_default_off_and_requires_external_contract():
    assert not TokenIdCaptureConfig().token_id_capture.fail_fast_poisoned_rollouts
    with pytest.raises(ValueError,match='requires external_staging'):
        TokenIdCaptureConfig(token_id_capture={'fail_fast_poisoned_rollouts':True})
