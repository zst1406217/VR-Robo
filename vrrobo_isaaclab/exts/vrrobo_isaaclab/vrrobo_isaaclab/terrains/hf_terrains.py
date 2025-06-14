# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

import scipy.interpolate as interpolate

from .utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def perlin_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPerlinTerrainCfg) -> np.ndarray:
    """Generate a height field using Perlin noise."""
    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)

    heightsamples_float = generate_fractal_noise_2d(
        cfg.size[0], cfg.size[1], width_pixels, length_pixels, frequency=cfg.frequency, zScale=cfg.zScale * difficulty
    )
    heightsamples_float -= heightsamples_float.mean()
    heightsamples = (heightsamples_float * (1 / cfg.vertical_scale)).astype(np.int16)
    heightsamples[0, :] = 0
    heightsamples[-1, :] = 0
    heightsamples[:, 0] = 0
    heightsamples[:, -1] = 0
    return heightsamples


@staticmethod
def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5


@staticmethod
def generate_fractal_noise_2d(
    xSize=20,
    ySize=20,
    xSamples=1600,
    ySamples=1600,
    frequency=10,
    fractalOctaves=2,
    fractalLacunarity=2.0,
    fractalGain=0.25,
    zScale=0.23,
):
    # zScale=0
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1
    shape = (xSamples, ySamples)
    noise = np.zeros(shape)
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d((xSamples, ySamples), (xScale, yScale)) * zScale
        amplitude *= fractalGain
        xScale, yScale = int(fractalLacunarity * xScale), int(fractalLacunarity * yScale)

    return noise
