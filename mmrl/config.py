"""Compose Gym services without changing the trainer's dependency lock."""

import copy
from dataclasses import asdict
from pathlib import Path

from .catalog import HARNESS_NAMES
from .process_rewards import ProcessRewardConfig
from .sandbox import validate_endpoint, validate_image


def validate_capture_retries(cfg):
    attempts = cfg.get("async_rl", {}).get("rollout_failure", {}).get(
        "nemo_gym", {}
    ).get("max_row_attempts")
    if type(attempts) is not int or attempts != 1:
        raise ValueError(
            "Capture requires async_rl.rollout_failure.nemo_gym.max_row_attempts=1; "
            "the recovery ledger owns retries with fresh physical attempt IDs"
        )


def validate_thinking_parser(cfg):
    """Validate the composed training recipe before credentials or actor setup.

    Missing enable_thinking leaves legacy/non-thinking recipes unchanged. This
    checks configuration consistency, not parser availability or model parity.
    Error text contains schema paths only, never supplied configuration values.
    """
    def mapping(value, path):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        return value

    def descend(value, keys, *, optional=False):
        path = "config"
        value = mapping(value, path)
        for key in keys:
            path += "." + key
            if key not in value:
                if optional:
                    return {}
                raise ValueError(f"Missing {path}")
            value = mapping(value[key], path)
        return value

    model = descend(cfg, ("env", "nemo_gym", "policy_model", "responses_api_models", "vllm_model"))
    extra = descend(model, ("extra_body",), optional=True)
    template = descend(extra, ("chat_template_kwargs",), optional=True)
    if "enable_thinking" not in template:
        return
    thinking = template["enable_thinking"]
    if type(thinking) is not bool:
        raise ValueError("vllm_model.extra_body.chat_template_kwargs.enable_thinking must be a boolean")
    if not thinking:
        return
    serving = descend(cfg, ("policy", "generation", "vllm_cfg", "http_server_serving_chat_kwargs"))
    parser = serving.get("reasoning_parser")
    if not isinstance(parser, str) or not parser.strip() or "${" in parser:
        raise ValueError("Thinking-enabled Gym requires a resolved nonempty policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser")
    # The training Gym YAML supplies True; reject explicit false/null/nonboolean overrides.
    if model.get("uses_reasoning_parser", True) is not True:
        raise ValueError("Thinking-enabled Gym requires vllm_model.uses_reasoning_parser=true")


def validate_sibling_recipe(cfg):
    """Eight siblings share one GRPO group, independently of update batch size."""
    count = cfg.get("grpo", {}).get("num_generations_per_prompt")
    if type(count) is not int or count != 8:
        raise ValueError("Sibling-mixed requires grpo.num_generations_per_prompt=8")


def gym_config(
    image, gateway, concurrency=8, *, judge_concurrency=None,
    max_output_tokens=32768, context_window=65536, sandbox_memory_mib=16384,
    process_reward_config: ProcessRewardConfig | None = None,
    chartmimic_image: str | None = None,
    project_harness_adapter: bool = False,
):
    managed_port = 5999 if gateway == "auto" else None
    gateway = (
        f"https://${{mm_coding.resources_servers.mm_coding.host}}:{managed_port}"
        if managed_port is not None
        else validate_endpoint(gateway, "MMRL_CAPTURE_GATEWAY")
    )
    validate_image(image)
    if chartmimic_image is not None:
        validate_image(chartmimic_image)
    if isinstance(sandbox_memory_mib, bool) or not isinstance(sandbox_memory_mib, int) or sandbox_memory_mib < 1024:
        raise ValueError("Sandbox memory must be an integer of at least 1024 MiB")
    if judge_concurrency is None:
        judge_concurrency = concurrency
    for name, value in (("Harness", concurrency), ("Judge", judge_concurrency)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} concurrency must be a positive integer")
    if (
        any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (max_output_tokens, context_window)
        )
        or not 1 <= max_output_tokens <= context_window <= 262144
    ):
        raise ValueError("Require 1 <= output budget <= context window <= 262144")
    resource = {"type": "resources_servers", "name": "mm_coding"}
    model = {"type": "responses_api_models", "name": "policy_model"}
    result = {
        "model_endpoint_readiness_timeout_seconds": 1200,
        "mm_coding": {
            "resources_servers": {
                "mm_coding": {
                    "entrypoint": "app.py",
                    "domain": "coding",
                    "num_workers": 1,
                    "sandbox_image": image,
                    "sandbox_memory_mib": sandbox_memory_mib,
                    "judge_concurrency": judge_concurrency,
                    "capture_gateway_port": managed_port,
                }
            }
        },
        "mm_single": {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    "resources_server": resource,
                    "model_server": model,
                    "max_steps": 1,
                }
            }
        },
        "sandbox": {
            "opensandbox": {
                "connection": {
                    "domain": "${oc.env:OPENSANDBOX_DOMAIN}",
                    "api_key": "${oc.env:OPENSANDBOX_API_KEY}",
                    "protocol": "https",
                    "tls_verify": "${oc.decode:${oc.env:OPENSANDBOX_TLS_VERIFY,true}}",
                    "use_server_proxy": True,
                    "transport_backend": "aiohttp",
                },
                "operations": {"command_retries": 0},
            }
        },
    }
    if chartmimic_image is not None:
        result["mm_coding"]["resources_servers"]["mm_coding"]["chartmimic_image"] = chartmimic_image
    if process_reward_config is not None:
        if not isinstance(process_reward_config, ProcessRewardConfig):
            raise TypeError("process_reward_config must be a ProcessRewardConfig")
        if managed_port is None:
            raise ValueError("Process preview requires the managed capture gateway (--gateway auto)")
        service = result["mm_coding"]["resources_servers"]["mm_coding"]
        service["entrypoint"] = "process_app.py"
        service["process_reward_config"] = asdict(process_reward_config)
    if managed_port is not None:
        # Gym resolves interpolations before assigning per-service defaults.
        # NeMo RL supplies default_host from the actor's actual private IP.
        result["mm_coding"]["resources_servers"]["mm_coding"]["host"] = (
            "${default_host}"
        )
    settings = {
        "codex": {
            "codex_version": "0.144.4",
            "model": "policy",
            "openai_api_key": "${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}",
            "sandbox_mode": "danger-full-access",
            "timeout": 10800,
            "extra_config": {"model_context_window": context_window},
        },
        "opencode": {
            "opencode_version": "1.17.11",
            "model": "gym/policy",
            # The native model_server branch overwrites auth with EMPTY. The
            # outer HarnessAgent still supplies the scoped Gym capture URL.
            "model_server": None,
            "openai_base_url": "__SANDBOX_MODEL_URL__/v1",
            "openai_api_key": "${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}",
            "thinking": True,
            "context_window": context_window,
            "max_output_tokens": max_output_tokens,
            "env": {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(max_output_tokens)},
            "timeout": 10800,
            "opencode_config": {
                # Training replays exact history; do not replace it with a summary
                # or prune image observations. NeMo RL caps output to remaining context.
                "compaction": {"auto": False, "prune": False},
                "provider": {
                    "gym": {
                        "npm": "@ai-sdk/openai-compatible",
                        "options": {
                            "baseURL": "__SANDBOX_MODEL_URL__/v1",
                            "apiKey": "${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}",
                            # CLI lifetime alone does not extend provider HTTP
                            # deadlines. All three values use milliseconds.
                            "timeout": 10800000,
                            "headerTimeout": 10800000,
                            "chunkTimeout": 10800000,
                        },
                        "models": {
                            "policy": {
                                "attachment": True,
                                "reasoning": True,
                                "tool_call": True,
                                "interleaved": {"field": "reasoning"},
                                "modalities": {
                                    "input": ["text", "image"],
                                    "output": ["text"],
                                },
                                "limit": {
                                    "context": context_window,
                                    "output": max_output_tokens,
                                },
                            }
                        },
                    }
                },
                "permission": {
                    "bash": "allow",
                    "edit": "allow",
                    "webfetch": "deny",
                    "external_directory": {"/opt/mmrl/*": "allow", "/tmp/*": "allow"},
                },
            },
        },
        "hermes": {
            "terminal_backend": "local",
            "enabled_toolsets": ["terminal", "file", "vision"],
            "compression_enabled": False,
            "api_key": "${oc.env:MMRL_CAPTURE_GATEWAY_TOKEN}",
            "max_turns": 32,
            "temperature": 1.0,
            "max_tokens": max_output_tokens,
        },
    }
    for harness in HARNESS_NAMES:
        result[f"mm_{harness}"] = {
            "responses_api_agents": {
                ("mm_harness" if project_harness_adapter else "harness_agent"): {
                    "entrypoint": "app.py",
                    "agent": harness,
                    "resources_server": resource,
                    "model_server": model,
                    "sandbox_provider": "sandbox",
                    "sandbox_image": image,
                    "sandbox_python": "/opt/hermes/bin/python -m mmrl.hermes_runner"
                    if harness == "hermes"
                    else "/opt/gym/bin/python",
                    "sandbox_model_base_url": gateway.rstrip("/"),
                    "concurrency": concurrency,
                    "rollout_timeout": 11400,
                    "runner_failure_as_response": True,
                    "agent_kwargs": settings[harness],
                }
            }
        }
    return result


def training_config(base, gym_config_path, train_path, output_dir, *,
                    process_reward_config: ProcessRewardConfig | None = None,
                    balanced_eleven=False):
    """Keep model/optimizer/backend settings; replace task wiring explicitly."""
    cfg = copy.deepcopy(base)
    if not cfg.get("data_plane", {}).get("enabled") or "async_rl" not in cfg:
        raise ValueError("Supply a resolved V2 SingleController configuration")
    generation = cfg["policy"]["generation"]
    if generation["top_k"] not in (None, -1):
        raise ValueError("The native harness endpoint requires disabled top-k sampling")
    sampling = {
        "temperature": generation["temperature"],
        "top_p": generation["top_p"],
        "top_k": -1,
    }
    cfg.pop("defaults", None)
    cfg.pop("benchmark_validation", None)
    # Native V2 owns reconstruction and durable handoff. Never reuse a donor
    # run's capture store or ask Gym to rebuild the same trajectory a second time.
    capture_dir = str(Path(output_dir).resolve() / "captures")
    cfg["token_capture"] = {
        "enabled": True,
        "num_reassembler_workers": 2,
        "capture_dir": capture_dir,
    }
    if process_reward_config is not None:
        if not isinstance(process_reward_config, ProcessRewardConfig):
            raise TypeError("process_reward_config must be a ProcessRewardConfig")
        if cfg.get("grpo", {}).get("adv_estimator", {}).get("name") != "grpo":
            raise ValueError("Process rewards require GRPO")
        cfg["token_capture"]["process_returns"] = {
            "binder": "mmrl.process_capture.bind_native_returns",
            "advantage": "mmrl.process_advantages.compute_process_advantages",
        }
    # The inherited probe uses the old validation loader, not captured Gym turns.
    cfg.setdefault("model_parity", {})["enabled"] = False
    inherited_train = copy.deepcopy(cfg["data"].get("train", {}))
    cfg["data"] = {
        "shuffle": True,
        "num_workers": 0,
        "use_multiple_dataloader": False,
        "max_input_seq_length": cfg["data"]["max_input_seq_length"],
        "train": {**inherited_train, "data_path": str(Path(train_path).resolve())},
        "default": {
            "dataset_name": "NemoGymDataset",
            "env_name": "nemo_gym",
            "prompt_file": None,
            "system_prompt_file": None,
            "processor": "nemo_gym_data_processor",
        },
    }
    cfg["env"] = {
        "should_use_nemo_gym": True,
        "should_mask_flagged_samples": True,
        "should_log_nemo_gym_responses": True,
        "nemo_gym": {
            "uv_venv_dir": str(Path(output_dir).resolve() / "gym-venvs"),
            "skip_venv_if_present": True,
            "pad_dynamic_image_shapes": True,
            "config_paths": [
                "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                str(Path(gym_config_path).resolve()),
            ],
            "token_id_capture": {
                "enabled": True,
                "all_agents": True,
                "rebuild_response": False,
                "max_mask_fraction": 0.05,
                "mask_fraction_min_samples": 1,
                "dir": capture_dir,
            },
            "policy_model": {
                "responses_api_models": {
                    "vllm_model": {
                        "request_prompt_and_generation_token_ids": True,
                        # Native Gym applies this last on every engine request,
                        # including requests issued directly by external CLIs.
                        "sampling_overrides": sampling,
                        "extra_body": {
                            "chat_template_kwargs": {"enable_thinking": True}
                        },
                    }
                }
            },
        },
    }
    if balanced_eleven:
        from .balancing import balanced_indices

        seed = cfg["grpo"].get("seed", 42)
        batch_size = cfg["grpo"]["num_prompts_per_step"]
        with Path(train_path).open() as source:
            balanced_indices(source, seed=seed, batch_size=batch_size)
        cfg["data"]["train"].update(
            dataset_name="mmrl.balanced_dataset.ElevenBenchmarkGymDataset",
            seed=seed,
            balance_batch_size=batch_size,
        )
    # Validation data must be provided separately; do not silently reuse a stale task loader.
    cfg["grpo"].update(
        val_at_start=False, val_at_end=False, val_period=0, max_rollout_turns=1
    )
    cfg["checkpointing"]["checkpoint_dir"] = str(
        Path(output_dir).resolve() / "checkpoints"
    )
    cfg["logger"]["log_dir"] = str(Path(output_dir).resolve() / "logs")
    cfg["logger"]["wandb_enabled"] = False
    cfg["async_rl"].setdefault("stall_watchdog", {})["stall_timeout_s"] = 12600.0
    failure = cfg["async_rl"].setdefault("rollout_failure", {})
    failure.pop("native", None)
    failure.setdefault("nemo_gym", {}).update(
        rollout_timeout_s=12000,
        # Inner Gym retries reuse capture IDs. The recovery ledger must own
        # retries so unfinished siblings get fresh physical attempt IDs.
        max_row_attempts=1,
    )
    return cfg
