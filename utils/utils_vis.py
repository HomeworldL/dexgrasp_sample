import numpy as np
import time
from typing import Dict, Any, Tuple, Union, Sequence, Optional
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


def visualize_with_plotly(
    meshes: Optional[Dict[str, Union[trimesh.Trimesh, Dict[str, Any]]]] = None,
    pointclouds: Optional[Dict[str, Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]]] = None,
    frames: Optional[np.ndarray] = None,
    camera_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    default_point_size: float = 3.0,
    axis_length: float = 0.03,
    show: bool = True,
) -> "plotly.graph_objects.Figure":
    """
    Plot meshes, point clouds and frame axes with Plotly.

    Frame format is (N,7): [px, py, pz, qw, qx, qy, qz].
    """
    try:
        import plotly.graph_objects as go
        from scipy.spatial.transform import Rotation as R
    except Exception as exc:
        raise RuntimeError("plotly/scipy is required for visualize_with_plotly.") from exc

    meshes = meshes or {}
    pointclouds = pointclouds or {}

    def _to_css_rgb(rgb: np.ndarray) -> str:
        arr = np.asarray(rgb)
        if arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0, 0, 255).astype(int)
        else:
            arr = np.clip(arr, 0, 255).astype(int)
        return f"rgb({arr[0]},{arr[1]},{arr[2]})"

    def _mesh_mean_color(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
        try:
            vc = None
            if hasattr(mesh, "visual") and getattr(mesh.visual, "vertex_colors", None) is not None:
                vc = np.asarray(mesh.visual.vertex_colors)
            elif hasattr(mesh, "visual") and getattr(mesh.visual, "face_colors", None) is not None:
                vc = np.asarray(mesh.visual.face_colors)
            if vc is None or vc.size == 0:
                return None
            vc = vc[:, :3].astype(float)
            if vc.max() > 2.0:
                vc /= 255.0
            return np.clip(vc.mean(axis=0), 0.0, 1.0)
        except Exception:
            return None

    fig = go.Figure()

    for name, val in meshes.items():
        mesh = val.get("mesh") if isinstance(val, dict) else val
        visible = bool(val.get("visible", True)) if isinstance(val, dict) else True
        if not visible or mesh is None:
            continue
        try:
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces, dtype=int)
            if vertices.size == 0 or faces.size == 0:
                continue
            mean_color = _mesh_mean_color(mesh)
            mesh_color = _to_css_rgb(mean_color) if mean_color is not None else "lightblue"
            fig.add_trace(
                go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.70,
                    color=mesh_color,
                    name=name,
                    flatshading=True,
                    hoverinfo="name",
                )
            )
        except Exception as exc:
            print(f"[plotly_vis] failed to add mesh {name}: {exc}")

    for name, val in pointclouds.items():
        try:
            if isinstance(val, dict):
                points = np.asarray(val.get("points", []))
                colors = val.get("colors", None)
                point_size = float(val.get("point_size", default_point_size))
            else:
                points, colors = val
                points = np.asarray(points)
                point_size = default_point_size
            if points.size == 0:
                continue
            if colors is None:
                color_list = ["rgb(200,200,200)"] * points.shape[0]
            else:
                colors = np.asarray(colors)
                if colors.ndim == 1 and colors.shape[0] == 3:
                    color_list = [_to_css_rgb(colors)] * points.shape[0]
                else:
                    if colors.max() > 2.0:
                        colors = (colors[:, :3] / 255.0).clip(0.0, 1.0)
                    else:
                        colors = colors[:, :3].clip(0.0, 1.0)
                    color_list = [_to_css_rgb(c) for c in colors]

            fig.add_trace(
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                    name=name,
                    marker=dict(size=point_size, color=color_list),
                )
            )
        except Exception as exc:
            print(f"[plotly_vis] failed to add pointcloud {name}: {exc}")

    if frames is not None:
        frames = np.asarray(frames)
        if frames.ndim == 2 and frames.shape[1] >= 7:
            origins = frames[:, :3]
            quats_wxyz = frames[:, 3:7]
            quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
            rots = R.from_quat(quats_xyzw)

            basis = np.array(
                [
                    [axis_length, 0.0, 0.0],
                    [0.0, axis_length, 0.0],
                    [0.0, 0.0, axis_length],
                ],
                dtype=float,
            )

            xs_x, ys_x, zs_x = [], [], []
            xs_y, ys_y, zs_y = [], [], []
            xs_z, ys_z, zs_z = [], [], []
            oxs, oys, ozs = [], [], []
            for idx, p in enumerate(origins):
                dirs = rots[idx].apply(basis)
                xs_x += [p[0], p[0] + dirs[0, 0], None]
                ys_x += [p[1], p[1] + dirs[0, 1], None]
                zs_x += [p[2], p[2] + dirs[0, 2], None]
                xs_y += [p[0], p[0] + dirs[1, 0], None]
                ys_y += [p[1], p[1] + dirs[1, 1], None]
                zs_y += [p[2], p[2] + dirs[1, 2], None]
                xs_z += [p[0], p[0] + dirs[2, 0], None]
                ys_z += [p[1], p[1] + dirs[2, 1], None]
                zs_z += [p[2], p[2] + dirs[2, 2], None]
                oxs.append(p[0])
                oys.append(p[1])
                ozs.append(p[2])

            fig.add_trace(
                go.Scatter3d(
                    x=xs_x,
                    y=ys_x,
                    z=zs_x,
                    mode="lines",
                    line=dict(color="red", width=3),
                    name="frame_x",
                    hoverinfo="none",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=xs_y,
                    y=ys_y,
                    z=zs_y,
                    mode="lines",
                    line=dict(color="green", width=3),
                    name="frame_y",
                    hoverinfo="none",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=xs_z,
                    y=ys_z,
                    z=zs_z,
                    mode="lines",
                    line=dict(color="blue", width=3),
                    name="frame_z",
                    hoverinfo="none",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=oxs,
                    y=oys,
                    z=ozs,
                    mode="markers",
                    marker=dict(size=1, color="black"),
                    name="frame_origins",
                )
            )
        else:
            print("[plotly_vis] frames should be an (N,7) array of [px,py,pz,qw,qx,qy,qz]")

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            aspectmode="data",
            camera=dict(eye=dict(x=float(camera_position[0]), y=float(camera_position[1]), z=float(camera_position[2]))),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    if show:
        fig.show()
    return fig
