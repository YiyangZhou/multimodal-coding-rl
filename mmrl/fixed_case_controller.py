"""Held-out media at a refitted, non-colocated controller boundary."""
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4



def log_cases(logger, report, directory, step):
    """Keep complete media and answers; failures remain visible in the table."""
    import wandb
    from nemo_rl.utils.logger import WandbLogger

    media, table = {}, []
    counters = {'manimbench_v1': 0, '3dcodebench': 0}
    for i, case in enumerate(report['cases']):
        benchmark = case['benchmark']
        counters[benchmark] += 1
        family = 'animation' if benchmark == 'manimbench_v1' else '3d'
        key = f'{family}_{counters[benchmark]}'
        answer_path = Path(directory)/f'case-{i:02d}'/'answer.txt'
        answer = answer_path.read_text() if answer_path.exists() else ''
        reference = None
        if case.get('audit_path'):
            audit = Path(case['audit_path'])
            ref = audit/'reference_render.png'
            if ref.exists():
                reference = wandb.Image(str(ref))
                media[key+'_reference'] = reference
            metadata_name = 'animation.json' if family == 'animation' else 'object_views.json'
            metadata_path = audit/metadata_name
            if metadata_path.exists():
                capture = json.loads(metadata_path.read_text()).get('full_media_capture') or {}
                if capture.get('complete'):
                    files = capture['files']
                    for j, artifact in enumerate(files):
                        path = artifact['path']
                        if Path(path).suffix == '.mp4': value = wandb.Video(path, format='mp4')
                        elif Path(path).suffix == '.glb': value = wandb.Object3D(path)
                        else: value = wandb.Image(path)
                        media[key if j == 0 else f'{key}_scene_{j+1}'] = value
            ref_metadata = audit/('reference_animation.json' if family=='animation' else 'reference_object_views.json')
            if ref_metadata.exists():
                capture=json.loads(ref_metadata.read_text()).get('full_media_capture') or {}
                if capture.get('complete'):
                    for j, artifact in enumerate(capture['files']):
                        path=artifact['path']
                        if Path(path).suffix=='.mp4': media[f'{key}_reference_scene_{j+1}']=wandb.Video(path,format='mp4')
                        elif Path(path).suffix=='.glb': media[f'{key}_reference_model']=wandb.Object3D(path)
        table.append([case['task_id'], case['weight_version'], case['seed'], case['prompt'],
                      reference, answer, case['status'], case.get('finish_reason'),
                      case.get('result',{}).get('reward'), case.get('result',{}).get('execution')])
    for backend in logger.loggers:
        if isinstance(backend, WandbLogger):
            backend.log_metrics(media,step=step,prefix='Fixed Cases')
            backend.log_table(['task','policy_version','seed','prompt','reference','full_answer','status','finish_reason','raw_reward','execution'],table,step,'Fixed Cases/details')


async def run_fixed_cases(controller, *, at_start=False):
    config_path = os.environ.get('MMRL_FIXED_CASE_CONFIG')
    if not config_path:
        return
    config = json.loads(Path(config_path).read_text())
    step = controller._train_steps
    if not at_start and step % config['period'] != 0 and step != config['target_step']:
        return
    if controller._gen.blocks_training():
        raise ValueError('Fixed-case adapter requires non-colocated generation')
    controller._rollout_permitted.clear()
    try:
        active = tuple(controller._dispatched_rollouts)
        if active:
            await asyncio.gather(*active)
        endpoints = [u for u in controller._gen.dp_openai_server_base_urls if u]
        destination = Path(config['output'])/f'step-{step}'/uuid4().hex
        report = await run_case_process(config, endpoints, destination, controller._trainer_version)
        await asyncio.to_thread(log_cases,controller._logger,report,destination,step)
    finally:
        controller._rollout_permitted.set()


async def run_case_process(config, endpoints, destination, weight_version):
    """Keep sandbox SDK dependencies in the prevalidated Gym environment."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    spec_path = destination.parent/(destination.name+'.request.json')
    spec_path.write_text(json.dumps({'config':config,'endpoints':endpoints,
                                    'destination':str(destination),'weight_version':weight_version}))
    with spec_path.with_suffix('.log').open('wb') as log:
        process = await asyncio.create_subprocess_exec(config['gym_python'], '-B', '-m',
                    'mmrl.fixed_case_worker',str(spec_path),stdout=log,stderr=log)
        try:
            status = await process.wait()
        except asyncio.CancelledError:
            process.terminate()
            await process.wait()
            raise
    if status != 0:
        raise RuntimeError(f'Fixed-case worker failed with exit code {status}')
    report = json.loads((destination/'result.json').read_text())
    if report['weight_version'] != weight_version or len(report['cases']) != 6:
        raise ValueError('Fixed-case worker result identity mismatch')
    return report
