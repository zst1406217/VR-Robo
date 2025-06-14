from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg

HEAD_POS = [0.332, 0.0, 0.00]


def reach_goal(
    env: ManagerBasedRLEnv, base_height, command_name, threshold, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
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

    reward = (torch.norm(head_pos_r - pos_goal, dim=1) < threshold).float()

    return reward


class goal_dis(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.dis = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # root_env = asset.data.root_pos_w - env.scene.env_origins
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
        dis = torch.norm(head_pos_r - pos_goal, dim=1)
        start_id = torch.where(self.dis == 0.0)
        reward = self.dis - dis
        reward[start_id] = 0.0
        self.dis = dis
        return reward * (dis > 0.25)

    def reset(self, env_ids: torch.Tensor):
        self.dis[env_ids] = 0.0


class goal_dis_z(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        self.dis = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self, env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        root_env = asset.data.root_pos_w - env.scene.env_origins

        rgb_commands = env.command_manager.get_command(command_name)
        goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
        goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
        goal_tensor[:, :, 2] += base_height
        selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
        pos_goal = selected_goal.sum(dim=1)
        dis = torch.abs(root_env[:, 2] - pos_goal[:, 2])
        start_id = torch.where(self.dis == 0.0)
        reward = self.dis - dis
        reward[start_id] = 0.0
        self.dis = dis
        return reward

    def reset(self, env_ids: torch.Tensor):
        self.dis[env_ids] = 0.0


def goal_heading_l1(
    env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    root_env = asset.data.root_pos_w - env.scene.env_origins
    rgb_commands = env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)

    root_quat = asset.data.root_quat_w

    # Compute direction to goal in yaw
    direction_to_goal = pos_goal[:, :2] - root_env[:, :2]
    goal_yaw = torch.atan2(direction_to_goal[:, 1], direction_to_goal[:, 0])

    # Convert quaternion to yaw
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    current_yaw = torch.atan2(t3, t4)

    # Calculate yaw error
    yaw_error = torch.abs(goal_yaw - current_yaw)
    yaw_error = torch.where(yaw_error > torch.pi, 2 * torch.pi - yaw_error, yaw_error)

    # Convert yaw error to linear reward
    linear_reward = 2 * yaw_error / torch.pi

    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos = torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    dis = torch.norm(head_pos_r - pos_goal, dim=1)

    return 1 - linear_reward * (dis > 0.25)


def stand_still_at_goal(
    env: ManagerBasedRLEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # root_env = asset.data.root_pos_w - env.scene.env_origins
    rgb_commands = env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)

    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w

    head_pos = torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    dis = torch.norm(head_pos_r - pos_goal, dim=1)
    return (dis < 0.25) * (1 / (torch.norm(env.action_manager.get_term("joint_pos").velocity_command, dim=1) + 0.4))


def track_lin_vel_xy_exp_command(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(
            env.action_manager.get_term("joint_pos").velocity_command[:, :2] - asset.data.root_lin_vel_b[:, :2]
        ),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp_command(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(
        env.action_manager.get_term("joint_pos").velocity_command[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)
