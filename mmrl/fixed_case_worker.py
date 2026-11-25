"""Run fixed-case sandbox scoring inside the existing Gym Python environment."""
import asyncio
import json
import os
import sys
from pathlib import Path
from .fixed_case_eval import evaluate_cases
from .reward import QualityJudge
from .sandbox import RemoteSandbox, provider_config
from nemo_gym.sandbox.providers.registry import create_provider
from .service import CodingService

async def main(path):
    spec=json.loads(Path(path).read_text())
    config=spec['config']
    rows=[json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    resources=json.loads(Path(config['gym_config']).read_text())['mm_coding']['resources_servers']['mm_coding']
    provider_options=provider_config()
    provider_options['opensandbox']['connection']['transport_backend']='httpx'
    sandbox=RemoteSandbox(resources['sandbox_image'], provider=create_provider(provider_options),
                          memory_mib=resources['sandbox_memory_mib'])
    try:
        await evaluate_cases(rows,spec['endpoints'],CodingService(sandbox,QualityJudge(concurrency=2)),
                             spec['destination'],weight_version=spec['weight_version'],
                             seed=config['seed'],max_tokens=config['max_tokens'],
                             model_name=config['generation_model'])
    finally:
        await sandbox.provider.aclose()

if __name__=='__main__':
    os.umask(0o077)
    request_path=sys.argv[1]
    sys.argv=sys.argv[:1]
    asyncio.run(main(request_path))
