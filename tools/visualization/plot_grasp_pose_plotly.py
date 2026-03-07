import os
import trimesh
from typing import Dict, Any, Tuple, Union, Sequence
from typing import Sequence, Optional
import numpy as np
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from src.dataset_objects import DatasetObjects
from src.mj_ho import MjHO, RobotKinematics
from utils.utils_vis import visualize_with_viser

def visualize_with_plot(
    meshes: Dict[str, Union[trimesh.Trimesh, Dict[str, Any]]] = None,
    pointclouds: Dict[
        str, Union[Tuple[np.ndarray, np.ndarray], Dict[str, Any]]
    ] = None,
    frames: np.ndarray = None,
    camera_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    default_point_size: float = 3.0,
    axis_length: float = 0.03,
    show: bool = True,
) -> "plotly.graph_objects.Figure":
    """
    Plot meshes, pointclouds and coordinate frames using plotly.

    - meshes: name -> trimesh.Trimesh  OR name -> {'mesh':trimesh, 'visible':bool}
      (will use mean vertex color if available, else gray)
    - pointclouds: name -> (points Nx3, colors Nx3) OR name -> {'points':..., 'colors':..., 'point_size':...}
    - frames: Nx7 array: [px,py,pz, qw,qx,qy,qz]  (注意：此处假定 quaternion 为 w,x,y,z)
    - 返回 plotly Figure（若 show=True 则会调用 fig.show()）
    """
    import plotly.graph_objects as go
    from scipy.spatial.transform import Rotation as R

    meshes = meshes or {}
    pointclouds = pointclouds or {}

    def _rgb_to_css(rgb):
        """rgb: length-3 float in 0..1 or int 0..255 -> 'rgb(r,g,b)'"""
        arr = np.asarray(rgb)
        if arr.dtype.kind in ("f",):
            arr = np.clip(arr * 255.0, 0, 255).astype(int)
        else:
            arr = np.clip(arr, 0, 255).astype(int)
        return f"rgb({arr[0]},{arr[1]},{arr[2]})"

    def _mean_mesh_color(m):
        """从 trimesh 里尽可能取得颜色（返回 None 或 3-length 0..1 float）"""
        try:
            vc = None
            # trimesh visual 存在 vertex_colors or face_colors
            if hasattr(m, "visual") and getattr(m.visual, "vertex_colors", None) is not None:
                vc = np.asarray(m.visual.vertex_colors)
            elif hasattr(m, "visual") and getattr(m.visual, "face_colors", None) is not None:
                vc = np.asarray(m.visual.face_colors)
            if vc is None or vc.size == 0:
                return None
            # vc may be Nx4 RGBA in 0..255 or 0..1 floats
            vc3 = vc[:, :3].astype(float)
            # detect scale
            if vc3.max() > 2.0:
                vc3 = vc3 / 255.0
            meanc = np.clip(vc3.mean(axis=0), 0.0, 1.0)
            return meanc
        except Exception:
            return None

    fig = go.Figure()

    # --- add meshes ---
    for name, val in meshes.items():
        # allow dict wrapper or direct trimesh
        if isinstance(val, dict):
            mesh = val.get("mesh", None)
            visible = bool(val.get("visible", True))
            if not visible or mesh is None:
                continue
        else:
            mesh = val

        try:
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces, dtype=int)
            if verts.size == 0 or faces.size == 0:
                continue

            meanc = _mean_mesh_color(mesh)
            color_arg = _rgb_to_css(meanc) if meanc is not None else "lightgray"

            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    opacity=0.7,
                    color="lightblue",
                    name=name,
                    flatshading=True,
                    hoverinfo="name",
                )
            )
        except Exception as e:
            # 忽略单个 mesh 的异常，不影响其它绘制
            print(f"[plotly_vis] failed to add mesh {name}: {e}")

    # --- add pointclouds ---
    for name, val in pointclouds.items():
        try:
            if isinstance(val, dict):
                pts = np.asarray(val.get("points", []))
                cols = val.get("colors", None)
                psize = float(val.get("point_size", default_point_size))
            else:
                pts, cols = val
                pts = np.asarray(pts)
                cols = np.asarray(cols) if cols is not None else None
                psize = default_point_size

            if pts is None or pts.size == 0:
                continue
            # normalize colors to 0..1 if possible
            if cols is None:
                color_list = ["rgb(200,200,200)"] * pts.shape[0]
            else:
                cols = np.asarray(cols)
                if cols.ndim == 1 and cols.size == 3:
                    # single color
                    color_list = [_rgb_to_css(cols)] * pts.shape[0]
                else:
                    # per-point colors Nx3 or Nx4
                    if cols.max() > 2.0:
                        cols_f = (cols[:, :3] / 255.0).clip(0.0, 1.0)
                    else:
                        cols_f = cols[:, :3].clip(0.0, 1.0)
                    color_list = [_rgb_to_css(c) for c in cols_f]

            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    name=name,
                    marker=dict(size=psize, color=color_list),
                )
            )
        except Exception as e:
            print(f"[plotly_vis] failed to add pointcloud {name}: {e}")

    # --- add coordinate frames ---
    if frames is not None:
        frames = np.asarray(frames)
        if frames.ndim == 2 and frames.shape[1] >= 7:
            origins = frames[:, :3]
            quats = frames[:, 3:7]  # assumed order w,x,y,z
            # convert to scipy order x,y,z,w
            quats_xyzw = quats[:, [1, 2, 3, 0]]
            R_batch = R.from_quat(quats_xyzw)

            # prepare concatenated line segments for each axis to reduce number of traces
            xs_x, ys_x, zs_x = [], [], []
            xs_y, ys_y, zs_y = [], [], []
            xs_z, ys_z, zs_z = [], [], []

            # optional small markers for origins
            oxs, oys, ozs = [], [], []

            # basis vectors scaled
            basis = np.array([axis_length * np.array([1.0, 0.0, 0.0]),
                              axis_length * np.array([0.0, 1.0, 0.0]),
                              axis_length * np.array([0.0, 0.0, 1.0])])

            for idx in range(len(origins)):
                p = origins[idx]
                # rotate basis
                try:
                    dirs = R_batch[idx].apply(basis)  # (3,3)
                except Exception:
                    dirs = R.from_quat(quats_xyzw[idx]).apply(basis)

                r=0.2
                # x axis
                xs_x += [p[0], p[0] + r * dirs[0, 0], None]
                ys_x += [p[1], p[1] + r * dirs[0, 1], None]
                zs_x += [p[2], p[2] + r * dirs[0, 2], None]
                # y axis
                xs_y += [p[0], p[0] + r * dirs[1, 0], None]
                ys_y += [p[1], p[1] + r * dirs[1, 1], None]
                zs_y += [p[2], p[2] + r * dirs[1, 2], None]
                # z axis
                xs_z += [p[0], p[0] + r * dirs[2, 0], None]
                ys_z += [p[1], p[1] + r * dirs[2, 1], None]
                zs_z += [p[2], p[2] + r * dirs[2, 2], None]

                oxs.append(p[0]); oys.append(p[1]); ozs.append(p[2])

            # three long traces (red, green, blue)
            width = 2.5
            fig.add_trace(
                go.Scatter3d(
                    x=xs_x,
                    y=ys_x,
                    z=zs_x,
                    mode="lines",
                    line=dict(color="red", width=width),
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
                    line=dict(color="green", width=width),
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
                    line=dict(color="blue", width=width),
                    name="frame_z",
                    hoverinfo="none",
                )
            )
            # origins as small markers
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

    # --- layout: 隐藏坐标轴与网格，仅显示几何 ---
    cam = dict(eye=dict(x=float(camera_position[0]), y=float(camera_position[1]), z=float(camera_position[2])))
    scene = dict(
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        zaxis=dict(visible=False, showgrid=False, zeroline=False),
        aspectmode="data",
        camera=cam,
        bgcolor="rgba(0,0,0,0)",   # 透明背景（或改为白色 "white"）
    )
    fig.update_layout(scene=scene, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      paper_bgcolor="white", plot_bgcolor="white")
    # fig.write_image("grasp_data_plotly.png", width=3840, height=2160, scale=4)

    if show:
        fig.show()
    return fig

ds = DatasetObjects("assets/ycb_datasets")

obj_name = "035_power_drill"
obj_info = ds.get_info(obj_name)

xml_path = os.path.join(
    os.path.dirname(__file__), "./assets/hands/liberhand/liberhand_right.xml"
)
mjho = MjHO(obj_info, xml_path)
rk = RobotKinematics(xml_path)

grasp_data_path = os.path.join(
    os.path.dirname(__file__), f"./outputs/{obj_name}/grasp_data.npy"
)
grasp_data = np.load(grasp_data_path, allow_pickle=True).item()
# print(f"Loaded grasp data: {grasp_data}")

qpos_init_list = grasp_data["qpos_init"]
qpos_approach_list=grasp_data["qpos_approach"]
qpos_prepared_list=grasp_data["qpos_prepared"]
qpos_grasp_list=grasp_data["qpos_grasp"]

# visualize
mesh_base = ds.get_mesh(obj_name, "inertia")

meshes_for_vis = {
    "mesh_base": mesh_base,
}

# frame_origin = np.array(qpos_init_list)[:, :3]
# pointclouds_for_vis = {
#     "frame_origin": (frame_origin, None),
# }

frame_for_vis = np.array(qpos_init_list)[:, :7]
visualize_with_plot(meshes_for_vis, None, frame_for_vis)