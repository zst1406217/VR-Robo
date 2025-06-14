#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import copy
import numpy as np
import pickle
import socket
import torch
from argparse import ArgumentParser
from torch import Tensor

import einops
import rpyc
from arguments import ModelParams, PipelineParams, get_combined_args
from e3nn import o3
from einops import einsum
from gaussian_renderer import GaussianModel, render
from pytorch3d.transforms import matrix_to_quaternion
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
import json



@torch.jit.script
def rotation_matrix_from_quaternion(quaternion):
    """Convert a batch of quaternions *(w, x, y, z)* into 3x3 rotation matrices."""
    q = quaternion
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Elements of the rotation matrix (row‑major)
    R = torch.stack(
        [
            torch.stack([1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q3 * q0, 2 * q1 * q3 + 2 * q2 * q0], dim=1),
            torch.stack([2 * q1 * q2 + 2 * q3 * q0, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q1 * q0], dim=1),
            torch.stack([2 * q1 * q3 - 2 * q2 * q0, 2 * q2 * q3 + 2 * q1 * q0, 1 - 2 * q1 * q1 - 2 * q2 * q2], dim=1),
        ],
        dim=1,
    )

    return R


def to_so3(R: torch.Tensor) -> torch.Tensor:
    U, _, Vt = torch.linalg.svd(R)
    R_orth = U @ Vt

    if torch.det(R_orth) < 0:
        U[..., -1] *= -1
        R_orth = U @ Vt

    return R_orth


def transform_gaussians(gaussians, T, scale: float):
    """
    Apply an in-place 4x4 similarity transform (rotation + translation + uniform scale)
    to a set of Gaussian primitives.
    """
    device = gaussians._xyz.device

    # 1. Update Gaussian centers -----------------------------------------------
    ones = torch.ones((gaussians._xyz.shape[0], 1), device=device)
    xyz_h = torch.cat([gaussians._xyz, ones], dim=1)
    transformed_xyz_h = torch.matmul(torch.tensor(T, device=device).float(), xyz_h.transpose(0, 1)).transpose(0, 1)
    gaussians._xyz = transformed_xyz_h[:, :3]

    # 2. Update isotropic scale (log‑space) -------------------------------------
    gaussians._scaling += np.log(scale)

    # 3. Update rotation (stored as quaternion) ---------------------------------
    rotation_norm = T[:3, :3] / scale
    rotation_matrix = gaussians.get_rotation_matrix()
    new_rotation = torch.matmul(torch.tensor(rotation_norm, device=device).float(), rotation_matrix)
    gaussians._rotation = matrix_to_quaternion(new_rotation)

    # 4. Rotate spherical harmonics (SH) coefficients ---------------------------
    shs_feat = gaussians._features_rest.cpu().double()
    shs_feat = transform_shs(shs_feat, rotation_norm)
    gaussians._features_rest = shs_feat.float().to(device)

    return gaussians


def transform_shs(shs_feat, rotation_matrix):
    """Rotate SH features up to order 3."""
    # switch axes: yzx -> xyz
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    permuted_rotation_matrix = np.linalg.inv(P) @ rotation_matrix @ P
    rotation_matrix_fix = to_so3(torch.from_numpy(permuted_rotation_matrix))
    rot_angles = o3._rotation.matrix_to_angles(rotation_matrix_fix)

    # Wigner‑D blocks -----------------------------------------------------------
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2])

    # rotation of the shs features
    ## order‑1 SH ---------------------------------------------------------------
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, "n shs_num rgb -> n rgb shs_num")
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        "... i j, ... j -> ... i",
    )
    one_degree_shs = einops.rearrange(one_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 0:3] = one_degree_shs

    ## order‑2 SH ---------------------------------------------------------------
    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, "n shs_num rgb -> n rgb shs_num")
    two_degree_shs = einsum(
        D_2,
        two_degree_shs,
        "... i j, ... j -> ... i",
    )
    two_degree_shs = einops.rearrange(two_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 3:8] = two_degree_shs

    ## order‑3 SH ---------------------------------------------------------------
    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, "n shs_num rgb -> n rgb shs_num")
    three_degree_shs = einsum(
        D_3,
        three_degree_shs,
        "... i j, ... j -> ... i",
    )
    three_degree_shs = einops.rearrange(three_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


class GSRenderer:
    """Convenience wrapper that loads four *Gaussian Splatting* point-clouds
    (environment + three coloured groups) and renders RGB images from arbitrary
    camera poses in simulation environment."""

    def __init__(self, pipeline: PipelineParams, data_dir: str = "vr-robo-dataset"):
        with torch.inference_mode():
            self.device = "cuda"
            self.pipeline = pipeline

            self.gaussians_env = GaussianModel(sh_degree=3)
            self.gaussians_red = GaussianModel(sh_degree=3)
            self.gaussians_green = GaussianModel(sh_degree=3)
            self.gaussians_blue = GaussianModel(sh_degree=3)

            self.gaussians_env.load_ply(f"{data_dir}/pcd/scene/point_cloud.ply")
            self.gaussians_red.load_ply(f"{data_dir}/pcd/red/point_cloud.ply")
            self.gaussians_green.load_ply(f"{data_dir}/pcd/green/point_cloud.ply")
            self.gaussians_blue.load_ply(f"{data_dir}/pcd/blue/point_cloud.ply")
            
            with open(f"{data_dir}/transform.json") as f:
                params = json.load(f)

            T_env = np.array(params["env"]["T"])
            scale_env = params["env"]["scale"]

            bounding_box_red = np.array(params["red"]["bounding_box"])
            width_red = np.array(params["red"]["width"])
            T_red = np.array(params["red"]["T"])
            scale_red = params["red"]["scale"]

            bounding_box_green = np.array(params["green"]["bounding_box"])
            width_green = np.array(params["green"]["width"])
            T_green = np.array(params["green"]["T"])
            scale_green = params["green"]["scale"]

            bounding_box_blue = np.array(params["blue"]["bounding_box"])
            width_blue = np.array(params["blue"]["width"])
            T_blue = np.array(params["blue"]["T"])
            scale_blue = params["blue"]["scale"]

            # ----- Filter + transform each coloured cloud --------------
            self.gaussians_red = filter_gaussians_within_bounding_box(self.gaussians_red, bounding_box_red, width_red)
            self.gaussians_green = filter_gaussians_within_bounding_box(
                self.gaussians_green, bounding_box_green, width_green
            )
            self.gaussians_blue = filter_gaussians_within_bounding_box(
                self.gaussians_blue, bounding_box_blue, width_blue
            )

            self.gaussians_env = transform_gaussians(self.gaussians_env, T_env, scale_env)
            self.gaussians_red = transform_gaussians(self.gaussians_red, T_red, scale_red)
            self.gaussians_green = transform_gaussians(self.gaussians_green, T_green, scale_green)
            self.gaussians_blue = transform_gaussians(self.gaussians_blue, T_blue, scale_blue)

            # Additional +90° around Z for green & blue (sim‑specific) ---------
            T_Z_90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            self.gaussians_green = transform_gaussians(self.gaussians_green, T_Z_90, 1.0)
            self.gaussians_blue = transform_gaussians(self.gaussians_blue, T_Z_90, 1.0)

            models = [self.gaussians_env, self.gaussians_red, self.gaussians_green, self.gaussians_blue]
            fields = ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]
            for field in fields:
                setattr(self.gaussians_env, field, torch.cat([getattr(m, field) for m in models], dim=0))

            counts = [m._xyz.shape[0] for m in models]
            self.start_indices = np.cumsum([0] + counts).tolist()[1:]

            # ----- Camera calibrition (adjust according to your own camera)-------------
            self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            self.width, self.height = 320, 180
            self.fovx, self.fovy   = 1.5701, 1.0260
            self.znear, self.zfar  = 0.01, 100.0


    def render(
        self,
        pos: Tensor = torch.tensor([[-2.0, -0.5, 0.5]]),
        ori: Tensor = torch.tensor([[0.5, -0.5, 0.5, -0.5]]),
        red_pos: Tensor = torch.tensor([[-0.2, -0.2, 0.36]]),
        green_pos: Tensor = torch.tensor([[-0.2, -0.6, 0.36]]),
        blue_pos: Tensor = torch.tensor([[-0.2, -1.0, 0.36]]),
    ) -> Tensor:
        with torch.inference_mode():
            # "ros" - forward axis: +Z - down axis +Y - right axis +X - Offset is applied in the ROS convention
            self.pos = copy.deepcopy(pos).to(self.device)
            self.ori = copy.deepcopy(ori).to(self.device)
            self.red_pos = copy.deepcopy(red_pos).to(self.device)
            self.green_pos = copy.deepcopy(green_pos).to(self.device)
            self.blue_pos = copy.deepcopy(blue_pos).to(self.device)

            num_poses = len(self.pos)
            rotation = rotation_matrix_from_quaternion(self.ori)

            T_sim = torch.zeros([num_poses, 4, 4], device=self.device)
            T_sim[:, :3, :3] = rotation
            T_sim[:, :3, 3] = self.pos
            T_sim[:, 3, 3] = 1
            T_sim = torch.inverse(T_sim)  # camera→world

            self.render_images = []

            # --- Iterate over views ------------------------------------
            for i in range(num_poses):
                world_view_transform = T_sim[i].transpose(0, 1)
                projection_matrix = (
                    getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy)
                    .transpose(0, 1)
                    .cuda()
                )
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

                custom_cam = MiniCam(
                    self.width,
                    self.height,
                    self.fovy,
                    self.fovx,
                    self.znear,
                    self.zfar,
                    world_view_transform,
                    full_proj_transform,
                )
                self.gaussians_env._xyz[self.start_indices[0] : self.start_indices[1]] += self.red_pos[i]
                self.gaussians_env._xyz[self.start_indices[1] : self.start_indices[2]] += self.green_pos[i]
                self.gaussians_env._xyz[self.start_indices[2] : self.start_indices[3]] += self.blue_pos[i]
                render_pkg = render(custom_cam, self.gaussians_env, self.pipeline, self.background)
                self.gaussians_env._xyz[self.start_indices[0] : self.start_indices[1]] -= self.red_pos[i]
                self.gaussians_env._xyz[self.start_indices[1] : self.start_indices[2]] -= self.green_pos[i]
                self.gaussians_env._xyz[self.start_indices[2] : self.start_indices[3]] -= self.blue_pos[i]
                rendering = render_pkg["render"]
                self.render_images.append(rendering)

            self.render_images = torch.stack(self.render_images)
            ndarr = self.render_images.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            ndarr = ndarr.reshape(ndarr.shape[0], -1)

            try:
                send_tensor(ndarr)  # user‑defined network helper
                print("Successfully sent tensor", ndarr.shape)
            except:
                print("Failed to send tensor")


def send_tensor(tensor, host="localhost", port=12345):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect((host, port))
    data = pickle.dumps(tensor.cpu().numpy())
    s.sendall(data)
    s.close()


class RenderService(rpyc.Service):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def exposed_render(self, pos, ori, red_pos, green_pos, blue_pos):
        return self.renderer.render(pos, ori, red_pos, green_pos, blue_pos)


def filter_gaussians_within_bounding_box(gaussians, bounding_box, width):
    """
    Keep only those Gaussians whose centres fall inside a *local* axis-aligned
    box after transforming them into the box's coordinate frame.

    The operation is **in-place**: all per-Gaussian attributes are masked.
    """
    device = gaussians._xyz.device
    bounding_box = np.linalg.inv(bounding_box)

    # Homogeneous coordinates of Gaussian centres
    ones = torch.ones((gaussians._xyz.shape[0], 1), device=device)
    xyz_h = torch.cat([gaussians._xyz, ones], dim=1)
    transformed_xyz_h = torch.matmul(
        torch.tensor(bounding_box, device=device).float(), xyz_h.transpose(0, 1)
    ).transpose(0, 1)
    transformed_xyz = transformed_xyz_h[:, :3]

    mask = (
        (transformed_xyz[:, 0] >= -width[0] / 2)
        & (transformed_xyz[:, 0] <= width[0] / 2)
        & (transformed_xyz[:, 1] >= -width[1] / 2)
        & (transformed_xyz[:, 1] <= width[1] / 2)
        & (transformed_xyz[:, 2] >= -width[2] / 2)
        & (transformed_xyz[:, 2] <= width[2] / 2)
    )

    # Apply mask to *all* per‑Gaussian attributes
    for attr in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
        if hasattr(gaussians, attr):
            setattr(gaussians, attr, getattr(gaussians, attr)[mask])

    return gaussians


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument(
        "--data_dir", type=str, default="vr-robo-dataset", help="Base directory for the point cloud data"
    )
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    args = get_combined_args(parser)
    renderer = GSRenderer(pipeline.extract(args), data_dir=args.data_dir)
    renderer.render()
    server = rpyc.ThreadedServer(
        RenderService(renderer), port=18861, protocol_config={"allow_pickle": True, "allow_public_attrs": True}
    )
    server.start()
