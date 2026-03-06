# src/object_model.py
"""
DatasetObjects

Manage a processed dataset folder (e.g., assets/YCB/ycb_datasets).
 - Build an ordered index of objects and their important files (inertia.obj, simplified.obj, urdf/xml, convex pieces).
 - Sample surface points with Open3D.
 - Load mesh via trimesh.
 - Generic external command runner.
 - SuperDec integration: compute superquadric decomposition for one object or batch for entire dataset.
 - Save SuperDec parameters into JSON.
 - Visualize mesh / sampled pointcloud / convex pieces / SuperDec meshes using viser (fallback open3d).

Code comments and docstrings are in English.
"""

from collections import OrderedDict
from pathlib import Path
import os
import json
import subprocess
import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import trimesh

# lazy-optional imports; we will raise helpful errors if missing at runtime
try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None

try:
    import viser  # type: ignore
except Exception:
    viser = None

# Try to import SuperDec dependencies (used only when computing superquadrics)
_superdec_available = False
try:
    import torch
    from omegaconf import OmegaConf
    from superdec.superdec import SuperDec
    from superdec.utils.predictions_handler import PredictionHandler
    # dataloader utilities used for normalize/denormalize
    from superdec.data.dataloader import normalize_points, denormalize_outdict, denormalize_points
    from superdec.data.transform import rotate_around_axis
    _superdec_available = True
except Exception:
    _superdec_available = False


class DatasetObjects:
    """
    Represent a dataset root (e.g. assets/YCB/ycb_datasets) and provide utilities.

    Attributes:
        root (str): absolute path to dataset root
        index (OrderedDict): map object_name -> object_info_dict
            object_info_dict keys: name, folder_abs, inertia_abs, simplified_abs, manifold_abs,
                                   coacd_abs, urdf_abs, xml_abs, meshes: {abs:[], rel:[]}, superdec: optional
    """

    def __init__(self, dataset_root: str):
        """
        Initialize dataset index by scanning immediate subfolders (ordered by leading number).
        """
        self.root = os.path.abspath(dataset_root)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.index: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._build_index()

        # SuperDec runtime members (initialized on demand)
        self._sd_model = None
        self._sd_device = None
        self._sd_configs = None
        self._sd_ckpt_path = None

    # -------------------------
    # Index building
    # -------------------------
    def _build_index(self) -> None:
        """Scan dataset root and fill self.index with ordered object entries."""
        entries = [d for d in sorted(os.listdir(self.root)) if os.path.isdir(os.path.join(self.root, d))]

        # sort by leading number if present
        def leading_num(s: str):
            import re
            m = re.match(r"^(\d+)", s)
            return int(m.group(1)) if m else float("inf")

        entries.sort(key=lambda s: (leading_num(s), s.lower()))

        for name in entries:
            folder = os.path.join(self.root, name)
            info = {"name": name, "folder_abs": os.path.abspath(folder)}

            # pick candidates for main files
            def _first_exists(*rels):
                for r in rels:
                    p = os.path.join(folder, r)
                    if os.path.exists(p):
                        return os.path.abspath(p)
                return None

            info["inertia_abs"] = _first_exists("inertia.obj")
            info["simplified_abs"] = _first_exists("simplified.obj")
            info["manifold_abs"] = _first_exists("manifold.obj")
            info["coacd_abs"] = _first_exists("coacd.obj", "decomposed.obj")
            info["urdf_abs"] = _first_exists(f"{name}.urdf")
            info["xml_abs"] = _first_exists(f"{name}.xml")
            # find meshes (convex pieces) under common subfolders
            meshes = []
            for md in ("meshes", "convex_parts", "parts"):
                mdp = os.path.join(folder, md)
                if os.path.isdir(mdp):
                    for fn in sorted(os.listdir(mdp)):
                        if fn.lower().endswith(".obj"):
                            meshes.append(os.path.abspath(os.path.join(mdp, fn)))
                    if meshes:
                        break
            info["meshes"] = {"abs": meshes, "rel": [os.path.relpath(p, folder).replace("\\", "/") for p in meshes]}
            # placeholder for superdec results
            info["superdec"] = None
            # attach metadata if present (metadata.json or dataset_metadata.json)
            meta = None
            for mname in ("metadata.json", "dataset_metadata.json"):
                mp = os.path.join(folder, mname)
                if os.path.exists(mp):
                    try:
                        with open(mp, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception:
                        meta = None
                    break
            info["metadata"] = meta
            self.index[name] = info

    def get_index(self) -> OrderedDict:
        """Return the internal ordered index."""
        return self.index

    # -------------------------
    # Mesh loading
    # -------------------------
    def load_mesh(self, mesh_path: Optional[str] = None) -> trimesh.Trimesh:
        """
        Load a mesh via trimesh. If mesh_path is None, prefer inertia_abs, then simplified_abs.
        Return a Trimesh (scene concatenated if necessary).
        """
        mp = mesh_path
        if mp is None:
            # find first available per dataset object? but for load_mesh we expect a path or operate on a single object
            raise ValueError("mesh_path must be provided to load_mesh")
        mesh = trimesh.load(mp, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()]) if hasattr(mesh, "geometry") else mesh
        return mesh

    # -------------------------
    # Surface sampling (Open3D)
    # -------------------------
    def sample_surface_o3d(self, obj_path: str, n_points: int = 4096, method: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample surface points from an OBJ using Open3D.

        Returns:
            points (N,3) numpy array, normals (N,3) numpy array
        """
        if o3d is None:
            raise RuntimeError("open3d is required for sampling. Install open3d (`pip install open3d`).")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"OBJ path not found: {obj_path}")

        mesh_o3d = o3d.io.read_triangle_mesh(obj_path)
        if not mesh_o3d.has_triangles():
            raise RuntimeError(f"Loaded mesh has no triangles: {obj_path}")

        if not mesh_o3d.has_vertex_normals():
            mesh_o3d.compute_vertex_normals()

        if method == "uniform":
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
        elif method == "poisson":
            try:
                pcd = mesh_o3d.sample_points_poisson_disk(number_of_points=n_points, init_factor=5)
            except Exception:
                pcd = mesh_o3d.sample_points_uniformly(number_of_points=n_points)
        else:
            raise ValueError("Unknown sampling method: choose 'uniform' or 'poisson'")

        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_normals():
            norms = np.asarray(pcd.normals, dtype=np.float32)
        else:
            # fallback: zeros
            norms = np.zeros_like(pts)
        return pts, norms

    # -------------------------
    # External command runner
    # -------------------------
    def run_command(self, cmd: List[str], quiet: bool = False, check: bool = False) -> subprocess.CompletedProcess:
        """
        Run an external command list. Returns subprocess.CompletedProcess.
        If quiet=True, redirect stdout/stderr to DEVNULL.
        """
        if quiet:
            cp = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=check)
        else:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return cp

    # -------------------------
    # SuperDec integration
    # -------------------------
    def _init_superdec(self, checkpoints_folder: str, checkpoint_file: str, device: Optional[str] = None, config_name: str = "config.yaml", lm_optimization: bool = False):
        """
        Load SuperDec model into memory. Call once before using compute_superdec_*.
        """
        if not _superdec_available:
            raise RuntimeError("superdec environment not available: make sure `superdec` package and its dependencies are importable.")

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sd_device = device

        ckp_path = os.path.join(checkpoints_folder, checkpoint_file)
        cfg_path = os.path.join(checkpoints_folder, config_name)
        if not os.path.exists(ckp_path):
            raise FileNotFoundError(f"SuperDec checkpoint not found: {ckp_path}")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"SuperDec config not found: {cfg_path}")

        checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
        with open(cfg_path) as f:
            configs = OmegaConf.load(f)
        model = SuperDec(configs.superdec).to(device)
        model.lm_optimization = lm_optimization
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # keep config and model
        self._sd_model = model
        self._sd_ckpt_path = ckp_path
        self._sd_configs = configs

    def compute_superdec_single(self,
                                object_name: str,
                                checkpoints_folder: str,
                                checkpoint_file: str,
                                target_n_points: int = 4096,
                                z_up: bool = True,
                                normalize: bool = True,
                                lm_optimization: bool = False,
                                resolution: int = 30) -> Dict[str, Any]:
        """
        Compute SuperDec decomposition for a single object (by object_name key in index).
        Returns a dict with denormalized outdict arrays and optionally mesh/pcs saved.
        The result is stored in self.index[object_name]['superdec'].
        """
        if object_name not in self.index:
            raise KeyError(f"Object not found in dataset index: {object_name}")

        if not _superdec_available:
            raise RuntimeError("superdec integration not available. Ensure superdec package and dependencies installed.")

        # init model if not already
        if self._sd_model is None:
            self._init_superdec(checkpoints_folder, checkpoint_file, lm_optimization=lm_optimization)

        entry = self.index[object_name]
        # choose input mesh path: inertia preferred, else simplified, else manifold
        input_mesh = entry.get("inertia_abs") or entry.get("simplified_abs") or entry.get("manifold_abs")
        if not input_mesh or not os.path.exists(input_mesh):
            raise FileNotFoundError(f"No input mesh found for {object_name}")

        # sample points using Open3D
        pts, norms = self.sample_surface_o3d(input_mesh, n_points=target_n_points, method="uniform")
        # convert to numpy float32
        points_np = np.asarray(pts, dtype=np.float32)

        # normalization - use superdec's normalize_points
        if normalize:
            points_normed, translation, scale = normalize_points(points_np)
        else:
            points_normed = points_np.copy()
            translation = np.zeros(3)
            scale = 1.0

        if z_up:
            points_normed = rotate_around_axis(points_normed, axis=(1, 0, 0), angle=-np.pi/2, center_point=np.zeros(3))

        points_t = torch.from_numpy(points_normed).unsqueeze(0).to(self._sd_device).float()

        # forward
        with torch.no_grad():
            outdict = self._sd_model(points_t)
            # move tensors to cpu numpy
            for key in list(outdict.keys()):
                if isinstance(outdict[key], torch.Tensor):
                    outdict[key] = outdict[key].cpu().numpy()
            # denormalize
            translation_arr = np.array([translation])
            scale_arr = np.array([scale])
            outdict_den = denormalize_outdict(outdict, translation_arr, scale_arr, z_up)
            points_den = denormalize_points(points_t.cpu(), translation_arr, scale_arr, z_up)

        # create PredictionHandler to get mesh & segmented pcs
        pred_handler = PredictionHandler.from_outdict(outdict_den, points_den, [object_name])
        meshes = pred_handler.get_meshes(resolution=resolution)
        pcs = pred_handler.get_segmented_pcs()

        # store results in index
        sd_result = {
            "outdict": outdict_den,   # denormalized params (numpy arrays)
            "meshes": meshes,         # list of trimesh meshes (one per batch obj, here len==1)
            "pcs": pcs,               # list of open3d pointcloud-like objects
            "sampled_points": points_den.squeeze(0).numpy() if hasattr(points_den, "numpy") else np.array(points_den).squeeze(0),
            "timestamp": time.time(),
            "resolution": resolution
        }
        entry["superdec"] = sd_result

        # Optionally save mesh to file under object folder
        try:
            out_mesh_path = os.path.join(entry["folder_abs"], "superdec_mesh.obj")
            if meshes and meshes[0] is not None:
                meshes[0].export(out_mesh_path)
                sd_result["superdec_mesh_path"] = out_mesh_path
        except Exception:
            pass

        return sd_result

    def compute_superdec_all(self,
                             checkpoints_folder: str,
                             checkpoint_file: str,
                             target_n_points: int = 4096,
                             z_up: bool = True,
                             normalize: bool = True,
                             lm_optimization: bool = False,
                             resolution: int = 30,
                             skip_existing: bool = True) -> Dict[str, Any]:
        """
        Compute SuperDec for all objects in the dataset (sequentially).
        Stores results into each object's info under 'superdec'.
        Returns a dict mapping object_name -> result (same as compute_superdec_single returned).
        """
        results = {}
        # initialize once
        if not _superdec_available:
            raise RuntimeError("superdec integration not available. Ensure superdec package and dependencies installed.")
        if self._sd_model is None:
            self._init_superdec(checkpoints_folder, checkpoint_file, lm_optimization=lm_optimization)

        for name, info in self.index.items():
            if skip_existing and info.get("superdec") is not None:
                results[name] = info["superdec"]
                continue
            try:
                res = self.compute_superdec_single(name, checkpoints_folder, checkpoint_file,
                                                   target_n_points=target_n_points, z_up=z_up,
                                                   normalize=normalize, lm_optimization=lm_optimization,
                                                   resolution=resolution)
                results[name] = res
            except Exception as e:
                # store error info but continue
                results[name] = {"error": str(e)}
        return results

    # -------------------------
    # Save/load SuperDec parameters
    # -------------------------
    def save_superdec_params(self, out_json: str):
        """
        Collect superdec parameters from index and save into JSON at out_json.
        Each object entry will include a 'superdec' sub-dict with arrays converted to python lists.
        """
        out = {}
        for name, info in self.index.items():
            sd = info.get("superdec")
            if sd is None:
                out[name] = None
                continue
            out[name] = {}
            od = sd.get("outdict", {})
            # convert numpy arrays inside outdict to lists
            for k, v in od.items():
                try:
                    out[name][k] = np.asarray(v).tolist()
                except Exception:
                    out[name][k] = v
            # include simple metadata: mesh path if exists
            out[name]["mesh_path"] = sd.get("superdec_mesh_path")
            out[name]["timestamp"] = sd.get("timestamp")
        ensure_dir_for_file(out_json)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # -------------------------
    # Visualization
    # -------------------------
    def visualize_object(self,
                         object_name: str,
                         show_mesh: bool = True,
                         show_pcd: bool = True,
                         show_convex_pieces: bool = True,
                         show_superdec: bool = True,
                         point_size: float = 0.005):
        """
        Visualize a single object showing:
          - base mesh (inertia or simplified)
          - sampled point cloud (from superdec result if present)
          - convex pieces (each different color)
          - superdec mesh (if computed)

        This tries to use viser first; if not available, falls back to open3d.
        Default up-axis is Z.
        """
        if object_name not in self.index:
            raise KeyError(f"Object not found: {object_name}")
        info = self.index[object_name]
        folder = info["folder_abs"]

        # prepare base mesh
        mesh_path = info.get("inertia_abs") or info.get("simplified_abs") or info.get("manifold_abs")
        if mesh_path is None or not os.path.exists(mesh_path):
            raise FileNotFoundError(f"No mesh available to visualize for {object_name}")

        base_mesh = self.load_mesh(mesh_path)

        # prepare point cloud from superdec sampled points if exists
        pcd_pts = None
        if show_pcd and info.get("superdec") is not None:
            try:
                s_pts = info["superdec"].get("sampled_points")
                if s_pts is not None:
                    pcd_pts = np.asarray(s_pts)
            except Exception:
                pcd_pts = None

        # prepare convex pieces
        pieces = []
        if show_convex_pieces:
            for p in info["meshes"]["abs"]:
                try:
                    pm = trimesh.load(p, process=False)
                    if isinstance(pm, trimesh.Scene):
                        pm = trimesh.util.concatenate([g for g in pm.geometry.values()]) if hasattr(pm, "geometry") else pm
                    pieces.append(pm)
                except Exception:
                    continue

        # prepare superdec mesh
        sd_mesh = None
        if show_superdec and info.get("superdec") is not None:
            try:
                sres = info["superdec"]
                meshes = sres.get("meshes")
                if meshes and meshes[0] is not None:
                    sd_mesh = meshes[0]
            except Exception:
                sd_mesh = None

        # Use viser if available
        if viser is not None:
            try:
                server = viser.ViserServer()
                # add base mesh
                server.scene.add_mesh_trimesh("base_mesh", mesh=base_mesh, visible=True)
                # add convex pieces each colored differently
                for i, pm in enumerate(pieces):
                    color = [int(255 * c) for c in trimesh.visual.random_color().tolist()[:3]] if hasattr(trimesh.visual, "random_color") else [255, 128, 0]
                    server.scene.add_mesh_trimesh(f"piece_{i}", mesh=pm, visible=True)
                # add superdec mesh
                if sd_mesh is not None:
                    server.scene.add_mesh_trimesh("superdec", mesh=sd_mesh, visible=True)

                # add point clouds
                if pcd_pts is not None:
                    server.scene.add_point_cloud(name="sampled_pts", points=pcd_pts, colors=None, point_size=point_size)

                # set up axis: default z-up
                server.scene.set_up_direction([0.0, 0.0, 1.0])

                @server.on_client_connect
                def _(client: "viser.ClientHandle"):
                    client.camera.position = (0.8, 0.8, 0.8)
                    client.camera.look_at = (0.0, 0.0, 0.0)

                # block keep alive
                try:
                    while True:
                        time.sleep(10.0)
                except KeyboardInterrupt:
                    return
            except Exception:
                # fallthrough to open3d
                pass

        # Fallback to open3d
        if o3d is None:
            raise RuntimeError("Neither viser nor open3d available for visualization.")

        geometries = []
        # base mesh
        base_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(base_mesh.vertices),
                                             o3d.utility.Vector3iVector(base_mesh.faces))
        try:
            if hasattr(base_mesh.visual, "vertex_colors") and base_mesh.visual.vertex_colors is not None:
                vcols = np.asarray(base_mesh.visual.vertex_colors)[:, :3] / 255.0
                if vcols.shape[0] == len(base_mesh.vertices):
                    base_o3d.vertex_colors = o3d.utility.Vector3dVector(vcols)
        except Exception:
            pass
        geometries.append(base_o3d)

        # convex pieces
        for pm in pieces:
            po3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pm.vertices),
                                            o3d.utility.Vector3iVector(pm.faces))
            # color each piece randomly
            c = np.random.rand(3)
            try:
                po3.paint_uniform_color(c.tolist())
            except Exception:
                pass
            geometries.append(po3)

        # superdec mesh
        if sd_mesh is not None:
            sm_o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(sd_mesh.vertices),
                                              o3d.utility.Vector3iVector(sd_mesh.faces))
            try:
                sm_o3.paint_uniform_color([0.8, 0.2, 0.2])
            except Exception:
                pass
            geometries.append(sm_o3)

        # sampled pcd
        if pcd_pts is not None:
            pcd_o3 = o3d.geometry.PointCloud()
            pcd_o3.points = o3d.utility.Vector3dVector(pcd_pts)
            geometries.append(pcd_o3)

        o3d.visualization.draw_geometries(geometries)

# -------------------------
# small utility
# -------------------------
def ensure_dir_for_file(filepath: str):
    d = os.path.dirname(os.path.abspath(filepath))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -------------------------
# Example usage (for quick manual test)
# -------------------------
if __name__ == "__main__":
    # Quick demo: scan dataset, print index summary
    ds = DatasetObjects("assets/ycb_datasets")
    idx = ds.get_index()
    print("Found objects:", list(idx.keys()))
    # Example to compute superdec for one object (uncomment & set your checkpoints)
    ds._init_superdec("checkpoints/normalized", "ckpt.pt")
    res = ds.compute_superdec_single("002_master_chef_can", "checkpoints/normalized", "ckpt.pt")
    print("SuperDec result keys:", res.keys())
