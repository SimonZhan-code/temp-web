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


def get_env_cls(env_type, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "libero", "libero_ltl", "robotwin", "realworld")
        env_cfg: Optional environment configuration.

    Returns:
        Environment class corresponding to the environment type.
    """
    if env_type == "libero":
        from rlinf.envs.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == "libero_ltl":
        from rlinf.envs.libero.libero_ltl_env import LiberoLTLEnv

        return LiberoLTLEnv
    elif env_type == "libero_ltl_composition":
        from rlinf.envs.libero.libero_composition_env import LiberoCompositionEnv

        return LiberoCompositionEnv
    elif env_type == "robotwin":
        from rlinf.envs.robotwin.RoboTwin_env import RoboTwin

        return RoboTwin
    elif env_type == "realworld":
        from rlinf.envs.realworld.realworld_env import RealWorldEnv

        return RealWorldEnv
    elif env_type == "wan_wm":
        # Wan world-model rollout backend (file-IPC client; server runs in the
        # wanspike venv -- see tools/wan_spike/wan_rollout_server.py)
        from rlinf.envs.world_model.wan_client_env import WanClientEnv

        return WanClientEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
