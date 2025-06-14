from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.string as string_utils
import omni.log
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers.action_manager import ActionTerm

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from . import actions_cfg


class JointAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg: actions_cfg.JointActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self._num_joints, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._joint_names)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)


class VelocityCommandAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    """The configuration of the action term."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset

        policy_cfg = {
            "class_name": "ActorCriticRecurrent",
            "init_noise_std": 0.75,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        }

        actor_critic_class = eval(policy_cfg["class_name"])  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(48, 235, 12, **policy_cfg).to(
            self._env.device
        )
        policy_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.policy_dir))
        actor_critic.load_state_dict(torch.load(policy_dir)["model_state_dict"])
        actor_critic.eval()
        self.policy = actor_critic.act_inference
        self.velocity_range = torch.tensor(cfg.velocity_range).cuda()
        self.velocity_command = torch.zeros([self.num_envs, 3]).cuda()
        self.uplevel_frequency = cfg.uplevel_frequency
        self.lowlevel_counter = 0
        self.tanh = torch.nn.Tanh()

    def process_actions(self, actions: torch.Tensor):
        self.velocity_command = self.tanh(actions) * self.velocity_range

    def apply_actions(self):
        if self.lowlevel_counter % 4 == 0:
            obs = self.get_observations()
            with torch.inference_mode():
                obs[:, 9:12] = self.velocity_command
                joint_positions = self.policy(obs)
            super().process_actions(joint_positions)
        self.lowlevel_counter += 1
        self.lowlevel_counter %= 4
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def get_observations(self):
        """Returns the current observations of the environment."""
        obs = self._env.observation_manager.compute_group("locomotion")
        return obs

    @property
    def action_dim(self) -> int:
        return 3
