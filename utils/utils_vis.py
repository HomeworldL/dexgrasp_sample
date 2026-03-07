import numpy as np
import time
from typing import Dict, Any, Tuple, Union, Sequence
import trimesh
import colorsys
import random

try:
    import viser
except Exception:
    viser = None


def generate_ncolors(num: int) -> np.ndarray:
    """
    Generate `num` visually distinct RGB colors (uint8).
    """
    if num <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    # use HSL evenly spaced hues with some random jitter in S/L
    colors = []
    step = 360.0 / num
    for i in range(num):
        h = (i * step) % 360.0
        s = 0.8 + (random.random() * 0.2)  # saturation 0.8-1.0
        l = 0.45 + (random.random() * 0.1) # lightness 0.45-0.55
        r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.asarray(colors, dtype=np.uint8)

def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    """
    Normalize colors to floats in range [0,1] with shape (N,3).
    Accepts uint8 (0-255) or float (0-1).
    """
    if colors is None:
        return None
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must be (N,3) array or None.")
    if np.issubdtype(colors.dtype, np.integer):
        return (colors.astype(np.float32) / 255.0).clip(0.0, 1.0)
    else:
        # assume float
        return colors.astype(np.float32).clip(0.0, 1.0)


def visualize_with_viser(
    meshes: Dict[str, Union[trimesh.Trimesh, Dict[str, Any]]] = None,
    pointclouds: Dict[str, Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]] = None,
    frames: np.ndarray = None,
    up_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    camera_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    default_point_size: float = 0.001,
    blocking: bool = False,
) -> "viser.ViserServer":
    """
    Generic viser visualization helper.

    Args:
        meshes: dict mapping name -> trimesh.Trimesh or name -> {'mesh': trimesh, 'visible':bool, 'wireframe':bool}
        pointclouds: dict mapping name -> (points Nx3, colors Nx3) OR name -> {'points':..., 'colors':..., 'point_size':...}
        up_direction: scene up vector, default Z-up (0,0,1)
        camera_position: initial camera position (x,y,z)
        look_at: camera look-at target
        default_point_size: fallback point size for point clouds
        blocking: if True, function blocks in a keep-alive loop. Default False returns server immediately.

    Returns:
        viser.ViserServer instance
    """
    if viser is None:
        raise RuntimeError("viser is not installed in current environment.")
    meshes = meshes or {}
    pointclouds = pointclouds or {}

    server = viser.ViserServer()
    scene = server.scene

    # Add meshes
    for name, val in meshes.items():
        if isinstance(val, dict):
            mesh = val.get("mesh")
            visible = bool(val.get("visible", True))
            # add trimesh instance
            if mesh is None:
                continue
            # support wireframe flag by adding as lines (approx) or rely on trimesh visual settings
            # default: use add_mesh_trimesh
            try:
                scene.add_mesh_trimesh(name, mesh=mesh, visible=visible)
            except Exception as e:
                # fallback: try to export to vertices/faces
                try:
                    verts = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.faces)
                    scene.add_mesh(name, vertices=verts, faces=faces, visible=visible)
                except Exception:
                    print(f"[viser] failed to add mesh {name}: {e}")
        else:
            mesh = val
            try:
                scene.add_mesh_trimesh(name, mesh=mesh, visible=True)
            except Exception as e:
                try:
                    verts = np.asarray(mesh.vertices)
                    faces = np.asarray(mesh.faces)
                    scene.add_mesh(name, vertices=verts, faces=faces, visible=True)
                except Exception:
                    print(f"[viser] failed to add mesh {name}: {e}")

    # Add point clouds
    for name, val in pointclouds.items():
        if isinstance(val, dict):
            pts = np.asarray(val.get("points"))
            cols = val.get("colors", None)
            psize = val.get("point_size", default_point_size)
        else:
            # tuple expected (points, colors)
            pts, cols = val
            psize = default_point_size

        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        cols_norm = None
        if cols is not None:
            cols_norm = _normalize_colors(np.asarray(cols))
        else:
            # use a single gray color
            cols_norm = np.ones((pts.shape[0], 3), dtype=np.float32) * 0.8

        try:
            scene.add_point_cloud(
                name=name,
                points=pts,
                colors=cols_norm,
                point_size=float(psize),
                point_shape="circle",
            )
        except Exception as e:
            print(f"[viser] failed to add pointcloud {name}: {e}")

    if frames is not None:
        frames_arr = np.asarray(frames, dtype=np.float32)
        if frames_arr.ndim == 2 and frames_arr.shape[1] == 7:
            for i, row in enumerate(frames_arr):
                px, py, pz, w, x, y, z = map(float, row)
                name = f"frame_{i}"
                try:
                    # preferred API: add_frame(name, wxyz=(w,x,y,z), position=(px,py,pz))
                    scene.add_frame(name, wxyz=(w, x, y, z), position=(px, py, pz), axes_length=0.003, axes_radius=0.0005)
                except Exception:
                    # minimal fallback: try positional arguments
                    try:
                        scene.add_frame(name, (w, x, y, z), (px, py, pz))
                    except Exception:
                        print(f"[viser] failed to add frame {name}")
        else:
            print("[viser] frames should be an (N,7) array of [w,x,y,z,px,py,pz]")

    # set up direction (Viser expects a 3-vector; convert to list)
    scene.set_up_direction(list(up_direction))

    # camera setup on client connect
    @server.on_client_connect
    def _on_connect(client: "viser.ClientHandle") -> None:
        try:
            client.camera.position = tuple(camera_position)
            client.camera.look_at = tuple(look_at)
        except Exception:
            pass

    # Return server, optionally block
    if blocking:
        try:
            # keep alive, user can connect via UI
            while True:
                time.sleep(10.0)
        except KeyboardInterrupt:
            # allow ctrl-c to exit
            print("visualization stopped by user (KeyboardInterrupt).")
            return server
    else:
        return server
