import json
from types import SimpleNamespace
import httpx
import pytest
from fastapi import FastAPI,Request
from responses_api_models.vllm_model.app import VLLMModel,ClientResponseError
from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.token_id_capture.sink import CaptureContext,set_token_sink,reset_token_sink,resolve_parent,ExternalCaptureTerminalError,reject_external_capture_terminal
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.base_responses_api_model import SimpleResponsesAPIModel,install_model_call_capture,ModelCallCaptureConfig

async def invoke(*,external,empty,propagate,tmp_path):
    calls=[];store=FileLineageStore(tmp_path/'ledger')
    context=CaptureContext('r1','c1',None,lineage_store=store,external_staging=external)
    token=set_token_sink(context) if external is not None else None
    class Client:
        async def create_chat_completion(self,**kwargs):
            calls.append(1);error=ClientResponseError(None,(),status=400)
            error.response_content=b'{"error":{"message":"maximum context length exhausted SECRET_BODY","code":400}}'
            raise error
    try:
        items=[{'role':'user','content':'fixture'}]
        if empty:items.append({'role':'assistant','content':''})
        model=SimpleNamespace(config=SimpleNamespace(use_completions_api=False,sequential_reasoning_allowed=False,propagate_context_overflow_errors=propagate,model='fixture'),_preprocess_chat_completion_create_params=lambda request,body:body,_resolve_client=lambda request:Client())
        model._create_empty_chat_completion=lambda:VLLMModel._create_empty_chat_completion(model)
        try:response=await VLLMModel.chat_completions(model,SimpleNamespace(headers={}),NeMoGymChatCompletionCreateParamsNonStreaming(model='fixture',messages=items))
        except (ExternalCaptureTerminalError,ClientResponseError) as exc:response=exc
        return response,len(calls),await store.manifest('r1'),context
    finally:
        if token is not None:reset_token_sink(token)
        await store.close()

@pytest.mark.parametrize('empty',[False,True])
@pytest.mark.parametrize('propagate',[False,True])
async def test_external_no_inference_paths_are_typed_and_excluded(tmp_path,empty,propagate):
    response,calls,manifest,ctx=await invoke(external=True,empty=empty,propagate=propagate,tmp_path=tmp_path)
    assert isinstance(response,ExternalCaptureTerminalError)
    expected='empty_assistant_continuation' if empty else 'context_window_exhausted'
    assert response.code==expected
    assert response.category==('infrastructure_error' if empty else 'budget_exhausted')
    assert 'SECRET_BODY' not in str(response)
    assert calls==(0 if empty else 1)
    assert manifest['records']==[] and [x['reason'] for x in manifest['failures']]==[expected]
    assert ctx.external_commit_coords is None and not ctx.committed

@pytest.mark.parametrize('external',[None,False])
@pytest.mark.parametrize('empty',[False,True])
async def test_non_external_legacy_success_is_unchanged(tmp_path,external,empty):
    response,calls,manifest,ctx=await invoke(external=external,empty=empty,propagate=False,tmp_path=tmp_path)
    assert response.choices[0].finish_reason==('content_filter' if empty else 'length')
    assert manifest['records']==[] and manifest['failures']==[]

async def test_non_external_explicit_propagation_is_unchanged(tmp_path):
    response,_,manifest,_=await invoke(external=False,empty=False,propagate=True,tmp_path=tmp_path)
    assert isinstance(response,ClientResponseError) and response.status==400
    assert getattr(response,'nemo_gym_vllm_propagate_context_error') is True
    assert manifest['failures']==[]

@pytest.mark.parametrize('code,category',[('context_window_exhausted','budget_exhausted'),('empty_assistant_continuation','infrastructure_error')])
@pytest.mark.parametrize('stream',[False,True])
async def test_actual_middleware_and_dispatch_typed_terminal(tmp_path,monkeypatch,code,category,stream):
    store=FileLineageStore(tmp_path/'ledger')
    import nemo_gym.base_responses_api_model as api
    monkeypatch.setattr(api,'installed_lineage_store',lambda:store)
    monkeypatch.setenv('NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN','test-only')
    class Model:
        _invoke_responses=SimpleResponsesAPIModel._invoke_responses
        _stream_responses=SimpleResponsesAPIModel._stream_responses
        responses_dispatch=SimpleResponsesAPIModel.responses_dispatch
        async def responses(self,body):await reject_external_capture_terminal(code)
    model=Model();app=FastAPI()
    @app.post('/v1/responses')
    async def responses(request:Request):return await model.responses_dispatch(request,await request.json())
    install_model_call_capture(app,ModelCallCaptureConfig(),global_config_dict={'token_id_capture':{'enabled':True,'external_staging':True,'rebuild_response':False,'dir':str(tmp_path/'tokens')}})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        r=await client.post('/ng-rollout/r1/training-token-capture/v1/responses',json={'model':'policy','input':'fixture','stream':stream})
    if stream:
        assert r.status_code==200
        events=[json.loads(x[6:]) for x in r.text.splitlines() if x.startswith('data: ') and x[6:]!='[DONE]']
        terminal=next(x['response'] for x in events if x['type']=='response.failed')
        assert terminal['output']==[] and terminal['error']['code']==code
    else:
        assert r.status_code==400 and r.json()['error']['code']==code and r.json()['error']['type']==category
    manifest=await store.manifest('r1')
    assert manifest['records']==[] and manifest['failures'][0]['reason']==code
    await store.close()

async def test_prior_staged_custody_not_reattributed(tmp_path):
    from history_ledger_fixture import _record_call_1
    store=FileLineageStore(tmp_path/'ledger');await _record_call_1(store)
    before=(await store.manifest('r1'))['records']
    context=CaptureContext('r1','terminal-call',None,lineage_store=store,external_staging=True)
    token=set_token_sink(context)
    try:
        with pytest.raises(ExternalCaptureTerminalError):await reject_external_capture_terminal('context_window_exhausted')
        after=await store.manifest('r1')
        assert after['records']==before
        assert [(x['model_call_id'],x['reason']) for x in after['failures']]==[('terminal-call','context_window_exhausted')]
        assert context.external_commit_coords is None and not context.committed
    finally:reset_token_sink(token);await store.close()
