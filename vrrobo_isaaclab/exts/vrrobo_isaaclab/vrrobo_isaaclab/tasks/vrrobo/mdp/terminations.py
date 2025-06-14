# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""

HEAD_POS = [0.332, 0.0, 0.00]

# cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
# cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
# transform_visualizer = VisualizationMarkers(cfg)


def get_goal(
    env: ManagerBasedRLEnv, base_height, command_name, threshold, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    asset: Articulation = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos = torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)

    rgb_commands = env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height

    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)
    return torch.norm(head_pos_r - pos_goal, dim=1) < threshold
