from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import rpyc

rpyc.core.protocol.DEFAULT_CONFIG["allow_pickle"] = True
import atexit
import numpy as np
import os
import pickle
import socket
import threading
import torch.nn as nn

import omni.isaac.lab.utils.math as math_utils
import timm
import torchvision
import torchvision.models as models
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg
from PIL import Image
from torchvision import transforms

HEAD_POS = [0.332, 0.0, 0.00]


@torch.jit.script
def euler_to_quaternion(euler_angles):
    cy = torch.cos(euler_angles[:, 2] * 0.5)
    sy = torch.sin(euler_angles[:, 2] * 0.5)
    cp = torch.cos(euler_angles[:, 1] * 0.5)
    sp = torch.sin(euler_angles[:, 1] * 0.5)
    cr = torch.cos(euler_angles[:, 0] * 0.5)
    sr = torch.sin(euler_angles[:, 0] * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack((w, x, y, z), dim=-1)


class GSServer:
    def __init__(self, host="localhost", port=12345, time_out=10):
        self.host = host
        self.port = port
        self.time_out = time_out
        # self.data = None
        # self.last_data = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()

    def init_data(self, env_num):
        self.data = np.zeros((env_num, 3 * 180 * 320))
        self.last_data = np.zeros((env_num, 3 * 180 * 320))
        self.latency = np.random.randint(0, 2, size=(env_num, 1))
        self.env_num = env_num

    def receive_data(self, host="localhost", port=12345):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        conn, addr = s.accept()
        conn.settimeout(self.time_out)  # Set timeout for receiving data
        data = b""
        try:
            while True:
                packet = conn.recv(40960000)
                if not packet:
                    break
                data += packet
        except socket.timeout:
            print("No new tensor received for 10 seconds, terminating connection.")
        finally:
            conn.close()
        pickle_data = pickle.loads(data)
        return pickle_data

    def start(self):
        atexit.register(self.close)
        self.running = True
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def run(self):
        while self.running:
            data = self.receive_data(self.host, self.port)
            with self.lock:
                self.last_data = self.data
                self.data = data

    def get_data(self):
        with self.lock:
            is_start = (self.last_data == 0).all(axis=1).reshape(-1, 1)
            latency = self.latency
            return (latency * self.last_data + (1 - latency) * self.data) * (1 - is_start) + self.data * is_start

    def reset(self, env_ids):
        if env_ids is None:
            return
        else:
            env_ids = env_ids.cpu().numpy()
            self.data[env_ids] = np.zeros((len(env_ids), 3 * 180 * 320))
            self.last_data[env_ids] = np.zeros((len(env_ids), 3 * 180 * 320))
            self.latency[env_ids] = np.random.randint(0, 2, size=(len(env_ids), 1))


if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

import random
import threading


def base_lin_vel_zero(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return torch.zeros(env.num_envs, 3, device=env.device)


def base_pos_z_e(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_e = asset.data.root_pos_w - env.scene.env_origins
    return pos_e[:, 2].unsqueeze(-1)


def standing_velocity_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = env.command_manager.get_command(command_name)
    commands[:, :2] *= (torch.norm(commands[:, :2], dim=1) > 0.1).unsqueeze(1)
    commands[:, 2] *= torch.abs(commands[:, 2]) > 0.1
    return commands


def zero_velocity_commands(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = torch.zeros(env.num_envs, 3, device=env.device)
    return commands


def goal_pos(
    env: ManagerBasedEnv, base_height, command_name, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The goal position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    rgb_commands = env.command_manager.get_command(command_name)
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    selected_goal = goal_tensor * rgb_commands.unsqueeze(2)
    pos_goal = selected_goal.sum(dim=1)

    return pos_goal


def goal_pos_multi(
    env: ManagerBasedEnv, base_height, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The goal position in the environment frame."""
    goal_red = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_green = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_blue = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins
    goal_tensor = torch.stack([goal_red, goal_green, goal_blue], dim=1)
    goal_tensor[:, :, 2] += base_height
    pos_goal = goal_tensor.reshape(env.num_envs, -1)

    return pos_goal


def head_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    pos_r = asset.data.root_pos_w - env.scene.env_origins
    quat_r = asset.data.root_quat_w
    head_pos = torch.tensor(HEAD_POS, device=env.device).repeat(env.num_envs, 1)
    head_pos_r = pos_r + math_utils.quat_apply(quat_r, head_pos)
    return head_pos_r


class gs_image_feature(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.conn = rpyc.connect("localhost", 18861)
        self.image_server = GSServer()
        self.image_server.start()
        self.image_server.init_data(env.num_envs)
        self.encoder_model = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.encoder_model.head = nn.Identity()  # Remove the final fully connected layer
        self.encoder_model.to(env.device)
        self.encoder_model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01, 2.0))], p=0.8),
                transforms.RandomApply([transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.1)], p=0.1),
                transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]),
            ]
        )
        self.camera_pos_noise_scale = torch.tensor([0.01, 0.01, 0.01], device=env.device)
        self.camera_rot_noise_scale = torch.tensor([1.0, 1.0, 2.0], device=env.device)
        print("GS Server Initialized")
        self.save_count = 0

    def reset(self, env_ids: torch.Tensor | None = None):
        self.image_server.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        camera_pos: list = None,
        camera_rot: list = None,
        asset_offset_pos: list = None,
        asset_offset_rot: list = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        pos_r = asset.data.root_pos_w - env.scene.env_origins
        quat_r = asset.data.root_quat_w
        camera_pos = torch.tensor(camera_pos, device=env.device).repeat(env.num_envs, 1)
        camera_pos += (2 * torch.rand_like(camera_pos) - 1) * self.camera_pos_noise_scale
        camera_rot = torch.tensor(camera_rot, device=env.device).repeat(env.num_envs, 1)
        camera_rot += (2 * torch.rand_like(camera_rot) - 1) * self.camera_rot_noise_scale
        camera_rot = torch.deg2rad(camera_rot)
        camera_rot = euler_to_quaternion(camera_rot)

        asset_offset_pos = torch.tensor(asset_offset_pos, device=env.device).repeat(env.num_envs, 1)
        asset_offset_rot = torch.tensor(asset_offset_rot, device=env.device).repeat(env.num_envs, 1)
        camera_pos_r = pos_r + math_utils.quat_apply(quat_r, camera_pos) - asset_offset_pos
        camera_rot_r = math_utils.quat_mul(quat_r, camera_rot)
        camera_rot_r = math_utils.convert_camera_frame_orientation_convention(camera_rot_r, "world", "ros")
        # TODO: add asset_offset_rot
        red_cone = env.scene["cone_red"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        green_cone = env.scene["cone_green"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        blue_cone = env.scene["cone_blue"].data.object_pos_w.squeeze(1) - env.scene.env_origins - asset_offset_pos
        self.conn.root.render(camera_pos_r, camera_rot_r, red_cone, green_cone, blue_cone)
        images = torch.tensor(self.image_server.get_data()).float().to(env.device)
        images = images.reshape(env.num_envs, 3, 180, 320)
        if images is not None:
            ndarr = (images[0]).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            concat_im = Image.fromarray(ndarr)
            # concat_im.save('obs_test.png')

        images = self.preprocess(images / 255)
        with torch.inference_mode():
            image_feature = self.encoder_model(images)

        return image_feature


def rgb_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    commands = env.command_manager.get_command(command_name)
    return commands
