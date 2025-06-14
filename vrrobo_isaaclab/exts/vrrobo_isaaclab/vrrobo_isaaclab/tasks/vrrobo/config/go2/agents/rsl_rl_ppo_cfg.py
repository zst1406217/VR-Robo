# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from vrrobo_isaaclab.wrapper.rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2GSPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    device = "cuda:0"
    num_steps_per_env = 40
    max_iterations = 4000
    save_interval = 100
    experiment_name = "unitree_go2_gs"
    empirical_normalization = False
    resume = False
    load_run = "2025-01-11_20-23-09"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRecurrent",
        init_noise_std=0.9,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1,
    )
