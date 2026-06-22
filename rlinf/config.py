# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
from dataclasses import asdict
from enum import Enum
from typing import Optional, Union

import torch
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import HybridComponentPlacement

logging.getLogger().setLevel(logging.INFO)


class SupportedModel(Enum):
    # Embodied models
    OPENVLA = ("openvla", "embodied")
    OPENVLA_OFT = ("openvla_oft", "embodied")
    OPENPI = ("openpi", "embodied")
    MLP_POLICY = ("mlp_policy", "embodied")
    GR00T = ("gr00t", "embodied")
    CNN_POLICY = ("cnn_policy", "embodied")
    RANDOM_POLICY = ("random_policy", "embodied")

    def __new__(cls, value, category):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.category = category
        return obj


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


SUPPORTED_ROLLOUT_BACKENDS = []
SUPPORTED_TASK_TYPE = ["embodied", "sft"]
SUPPORTED_TRAINING_BACKENDS = ["fsdp"]
__all__ = ["build_config"]


def torch_dtype_from_precision(precision: Union[int, str]) -> torch.dtype:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )




def validate_fsdp_cfg(cfg: DictConfig, resume_dir: Optional[str] = None) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        if "amp" not in config:
            config.amp = {}
        config.amp.enabled = config.amp.get("enabled", False)
        config.amp.precision = config.amp.get("precision", "bf16")
        assert config.amp.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        config.amp.use_grad_scaler = config.amp.get("use_grad_scaler", False)
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        if resume_dir is not None:
            cfg.fsdp_config.use_orig_params = True

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )

        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", "bf16"
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", "bf16"
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", "fp32"
        )

    return cfg


def validate_embodied_cfg(cfg):
    assert get_supported_model(cfg.actor.model.model_type).category == "embodied", (
        f"Model type: '{cfg.actor.model.model_type}' is not an embodied model. "
        f"Supported embodied models: {[e.value for e in SupportedModel if e.category == 'embodied']}."
    )

    # NOTE: Currently we only support actor_critic as PPO algorithm loss, and only support value_head as critic model.
    # This will be updated in the future to support more algorithms and critic models.
    # Check that actor_critic / vmpo loss requires value_head (both train a value head).
    if cfg.algorithm.loss_type in ("actor_critic", "vmpo"):
        add_value_head = cfg.actor.model.get("add_value_head", False)
        assert add_value_head, (
            f"When using a value-based algorithm (algorithm.loss_type="
            f"'{cfg.algorithm.loss_type}'), actor.model.add_value_head must be True. "
            f"Current value: {add_value_head}"
        )

    # V-MPO is the single-critic on-policy method: it requires the algorithm.vmpo block.
    if cfg.algorithm.adv_type == "vmpo":
        assert cfg.algorithm.get("vmpo", None) is not None, (
            "algorithm.adv_type='vmpo' requires an algorithm.vmpo: config block."
        )

    # process num-envs
    component_placement = HybridComponentPlacement(
        cfg, Cluster(cluster_cfg=cfg.cluster)
    )
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")

    if cfg.runner.val_check_interval > 0 or cfg.runner.only_eval:
        assert cfg.env.eval.total_num_envs > 0, (
            "Total number of parallel environments for evaluation must be greater than 0"
        )
        assert cfg.env.eval.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes"
        )
        assert cfg.env.eval.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.eval.total_num_envs // env_world_size // stage_num > 0, (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.eval.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.eval.group_size
            == 0
        ), (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.eval.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.eval.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    if not cfg.runner.only_eval:
        assert cfg.env.train.total_num_envs > 0, (
            "Total number of parallel environments for training must be greater than 0"
        )
        assert cfg.env.train.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes"
        )
        assert cfg.env.train.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.train.total_num_envs // env_world_size // stage_num > 0, (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.train.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.train.group_size
            == 0
        ), (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.train.max_steps_per_rollout_epoch
            % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.train.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    return cfg


def validate_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    assert cfg.runner.task_type in SUPPORTED_TASK_TYPE, (
        f"task_type must be one of {SUPPORTED_TASK_TYPE}"
    )
    if cfg.runner.task_type == "embodied":
        cfg = validate_embodied_cfg(cfg)

    if cfg.algorithm.adv_type in ("grpo", "reinpp_baseline"):
        assert cfg.algorithm.group_size > 1

    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS, (
        f"Unsupported training_backend {cfg.actor.training_backend}. Supported training backends are {SUPPORTED_TRAINING_BACKENDS}."
    )

    component_placement = HybridComponentPlacement(
        cfg, Cluster(num_nodes=cfg.cluster.num_nodes)
    )
    actor_world_size = component_placement.get_world_size("actor")
    assert (
        cfg.actor.global_batch_size
        % (cfg.actor.micro_batch_size * actor_world_size)
        == 0
    ), (
        f"actor.global_batch_size ({cfg.actor.global_batch_size}) must be divisible by (actor.micro_batch_size ({cfg.actor.micro_batch_size}) * actor_world_size ({actor_world_size}))"
    )
    cfg.actor = validate_fsdp_cfg(cfg.actor, cfg.runner.get("resume_dir", None))

    if cfg.critic.use_critic_model:
        cfg.critic = validate_fsdp_cfg(cfg.critic)

    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)

