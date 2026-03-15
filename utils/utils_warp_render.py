import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

try:
    import pyglet

    pyglet.options["headless"] = True
except Exception:
    pyglet = None

try:
    import warp as wp
    import warp.render
except Exception:
    wp = None


def _np_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.clip(norm, eps, None)
    return vec / norm


def camera_spherical(sample_num: int, radius: float, rng: np.random.Generator) -> np.ndarray:
    points = rng.standard_normal((sample_num, 3)).astype(np.float64)
    return radius * _np_normalize(points)


def camera_circular_zaxis(sample_num: int, radius: float, center: Sequence[float]) -> np.ndarray:
    phi = np.pi * (3.0 - np.sqrt(5.0))
    theta = np.arange(sample_num, dtype=np.float64) * phi
    center = np.asarray(center, dtype=np.float64)
    offset = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    return center[None, :] + radius * offset


def camera_view_matrix(
    sample_num: int,
    pos: np.ndarray,
    rng: np.random.Generator,
    pos_noise: float = 0.0,
    min_radius: float = 0.0,
    min_radius_margin: float = 1e-3,
    lookat: Sequence[float] = (0.0, 0.0, 0.0),
    lookat_noise: float = 0.0,
    up: Optional[Sequence[float]] = None,
    up_noise: float = 0.0,
) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    pos = pos + pos_noise * (rng.random((sample_num, 3)) - 0.5)
    if min_radius > 0.0:
        pos_norm = np.linalg.norm(pos, axis=-1, keepdims=True)
        safe_radius = float(min_radius) + float(min_radius_margin)
        bad = pos_norm[:, 0] <= safe_radius
        if np.any(bad):
            pos_dir = _np_normalize(pos[bad])
            pos[bad] = pos_dir * safe_radius
    lookat = np.asarray(lookat, dtype=np.float64)
    lookat = lookat[None, :] + lookat_noise * (rng.random((sample_num, 3)) - 0.5)

    front = _np_normalize(lookat - pos)
    while True:
        if up is None:
            up_arr = rng.standard_normal((sample_num, 3))
        else:
            up_arr = np.asarray(up, dtype=np.float64)[None, :].repeat(sample_num, axis=0)
        up_arr = _np_normalize(up_arr + up_noise * (rng.random((sample_num, 3)) - 0.5))
        up_arr = up_arr - np.sum(up_arr * front, axis=-1, keepdims=True) * front
        up_arr = _np_normalize(up_arr)
        if not np.any(np.linalg.norm(up_arr, axis=-1) < 1e-6):
            break

    view_matrix = np.zeros((sample_num, 4, 4), dtype=np.float64)
    view_matrix[:, 0, :3] = np.cross(front, up_arr)
    view_matrix[:, 1, :3] = up_arr
    view_matrix[:, 2, :3] = -front
    view_matrix[:, 3, :3] = pos
    view_matrix[:, 3, 3] = 1.0
    return view_matrix


def get_camera_matrix(
    camera_cfg: Dict,
    sample_num: int,
    rng: np.random.Generator,
    min_radius: float = 0.0,
) -> np.ndarray:
    cam_type = str(camera_cfg["type"])
    if cam_type == "spherical":
        pos = camera_spherical(sample_num=sample_num, radius=float(camera_cfg["radius"]), rng=rng)
    elif cam_type == "circular_zaxis":
        pos = camera_circular_zaxis(
            sample_num=sample_num,
            radius=float(camera_cfg["radius"]),
            center=list(camera_cfg.get("center", [0.0, 0.0, 0.8])),
        )
    else:
        raise NotImplementedError(f"Unsupported camera type: {cam_type}")

    return camera_view_matrix(
        sample_num=sample_num,
        pos=pos,
        rng=rng,
        pos_noise=float(camera_cfg.get("pos_noise", 0.0)),
        min_radius=float(min_radius),
        lookat=list(camera_cfg.get("lookat", [0.0, 0.0, 0.0])),
        lookat_noise=float(camera_cfg.get("lookat_noise", 0.0)),
        up=camera_cfg.get("up", None),
        up_noise=float(camera_cfg.get("up_noise", 0.0)),
    )


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


class WarpPointCloudRenderer:
    def __init__(
        self,
        device: str,
        tile_width: int,
        tile_height: int,
        n_cols: int,
        n_rows: int,
        z_near: float,
        z_far: float,
        intrinsics: Intrinsics,
    ):
        if wp is None:
            raise RuntimeError("warp-lang is not available. Please install warp-lang.")

        self.device = device
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.n_cols = int(n_cols)
        self.n_rows = int(n_rows)
        self.num_tiles = self.n_cols * self.n_rows
        self.z_near = float(z_near)
        self.z_far = float(z_far)

        self.fx = float(intrinsics.fx)
        self.fy = float(intrinsics.fy)
        self.cx = float(intrinsics.cx)
        self.cy = float(intrinsics.cy)

        self.renderer = wp.render.OpenGLRenderer(
            draw_axis=False,
            draw_grid=False,
            show_info=False,
            draw_sky=False,
            screen_width=self.tile_width * self.n_cols,
            screen_height=self.tile_height * self.n_rows,
            near_plane=self.z_near,
            far_plane=self.z_far,
            vsync=False,
            headless=True,
        )

        proj = np.array(
            [
                [2.0 * self.fx / self.tile_width, 0.0, 0.0, 0.0],
                [0.0, 2.0 * self.fy / self.tile_height, 0.0, 0.0],
                [0.0, 0.0, -(self.z_far + self.z_near) / (self.z_far - self.z_near), -1.0],
                [0.0, 0.0, -2.0 * self.z_far * self.z_near / (self.z_far - self.z_near), 0.0],
            ],
            dtype=np.float64,
        )
        self.projection_matrices: List[np.ndarray] = [proj] * self.num_tiles

        self._setup_tile_done = False
        self._view_matrix_torch: Optional[torch.Tensor] = None

        yy, xx = torch.meshgrid(
            torch.arange(self.tile_height, device=self.device),
            torch.arange(self.tile_width, device=self.device),
            indexing="ij",
        )
        self._xx = xx[None, :, :, None]
        self._yy = yy[None, :, :, None]

    def update_camera_poses(self, view_matrix: np.ndarray) -> None:
        view = torch.as_tensor(view_matrix, dtype=torch.float32, device=self.device)
        self._view_matrix_torch = view
        inv_view = torch.inverse(view).cpu().numpy().tolist()

        if not self._setup_tile_done:
            self.renderer.setup_tiled_rendering(
                instances=[[0]] * self.num_tiles,
                tile_sizes=[(self.tile_width, self.tile_height)] * self.num_tiles,
                projection_matrices=self.projection_matrices,
                view_matrices=inv_view,
                tile_ncols=self.n_cols,
                tile_nrows=self.n_rows,
            )
            self._setup_tile_done = True
            return

        for tile_id in range(self.num_tiles):
            self.renderer.update_tile(
                tile_id=tile_id,
                instances=[0],
                tile_size=(self.tile_width, self.tile_height),
                projection_matrix=self.projection_matrices[tile_id],
                view_matrix=inv_view[tile_id],
            )

    def render_mesh(self, mesh: trimesh.Trimesh, view_matrix: np.ndarray) -> torch.Tensor:
        self.renderer.clear()
        self.update_camera_poses(view_matrix)

        self.renderer.begin_frame(self.renderer.clock_time)
        self.renderer.render_mesh(
            name="mesh",
            points=np.asarray(mesh.vertices, dtype=np.float32),
            indices=np.asarray(mesh.faces, dtype=np.int32),
        )
        self.renderer.end_frame()
        if self._view_matrix_torch is None:
            raise RuntimeError("Internal error: view matrix is not initialized.")
        return self._view_matrix_torch

    def get_image(self, mode: str = "depth") -> torch.Tensor:
        if mode == "depth":
            image = wp.zeros((self.num_tiles, self.tile_height, self.tile_width, 1), dtype=wp.float32)
        elif mode == "rgb":
            image = wp.zeros((self.num_tiles, self.tile_height, self.tile_width, 3), dtype=wp.float32)
        else:
            raise NotImplementedError(f"Unsupported image mode: {mode}")
        self.renderer.get_pixels(image, split_up_tiles=True, mode=mode)
        return wp.to_torch(image)

    def depth_to_world_point_cloud(self, depth_image: torch.Tensor) -> torch.Tensor:
        if self._view_matrix_torch is None:
            raise RuntimeError("Call render_mesh() before depth_to_world_point_cloud().")

        x = (self._xx - self.cx) * depth_image / self.fx
        y = -(self._yy - self.cy) * depth_image / self.fy
        camera = torch.cat([x, y, -depth_image, torch.ones_like(x, device=x.device)], dim=-1)
        world = camera.view(depth_image.shape[0], -1, 4) @ self._view_matrix_torch
        return world[..., :3]

    def depth_to_camera_point_cloud(self, depth_image: torch.Tensor) -> torch.Tensor:
        x = (self._xx - self.cx) * depth_image / self.fx
        y = -(self._yy - self.cy) * depth_image / self.fy
        z = -depth_image
        camera = torch.cat([x, y, z], dim=-1)
        return camera.view(depth_image.shape[0], -1, 3)


def intrinsics_from_config(cfg: Dict, tile_width: int, tile_height: int) -> Intrinsics:
    preset = str(cfg.get("preset", "kinect"))
    if preset == "kinect":
        fx = float(cfg.get("fx", 608.6939697265625))
        fy = float(cfg.get("fy", 608.6422119140625))
    else:
        fx = float(cfg["fx"])
        fy = float(cfg["fy"])

    return Intrinsics(
        fx=fx,
        fy=fy,
        cx=float(cfg.get("cx", tile_width // 2)),
        cy=float(cfg.get("cy", tile_height // 2)),
    )


def mesh_from_path(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return mesh
