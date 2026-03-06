import os
import torch
import numpy as np
from omegaconf import OmegaConf
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.data.dataloader import denormalize_outdict, denormalize_points, normalize_points
import open3d as o3d
import viser
from superdec.data.transform import rotate_around_axis
import time

def load_and_sample_pointcloud(path, target_n=4096, z_up=False, mesh_sample_method="uniform"):
    """
    Load either a point cloud file (ply, pc formats) or a mesh file (obj, stl, ply as mesh),
    and return an (N,3) numpy array of points (N == target_n).
    If loading mesh, will sample exactly target_n points from mesh surface.
    """
    ext = os.path.splitext(path)[1].lower()
    # If the file can be interpreted as a mesh (obj/stl/ply meshes), prefer mesh sampling
    mesh_exts = {'.obj', '.stl', '.ply', '.off', '.gltf', '.glb', '.fbx'}
    if ext in mesh_exts:
        try:
            # Try to read as mesh first
            mesh = o3d.io.read_triangle_mesh(path)
            if mesh is None or len(mesh.triangles) == 0:
                # fallback to read as point cloud
                pc = o3d.io.read_point_cloud(path)
                pts = np.asarray(pc.points)
            else:
                # Make sure mesh has vertex normals (some sampling methods can use them)
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                # Use Open3D's sampling (uniform or poisson)
                if mesh_sample_method == "poisson":
                    pcd = mesh.sample_points_poisson_disk(number_of_points=target_n, init_factor=5)
                else:
                    # uniform sampling will produce exactly target_n
                    pcd = mesh.sample_points_uniformly(number_of_points=target_n)
                pts = np.asarray(pcd.points)
        except Exception as e:
            # Fallback: try reading as point cloud
            pc = o3d.io.read_point_cloud(path)
            pts = np.asarray(pc.points)
    else:
        # treat as point cloud
        pc = o3d.io.read_point_cloud(path)
        pts = np.asarray(pc.points)

    # If sampled points are fewer/more than target_n, uniformly sample indices with or without replacement
    n_pts = pts.shape[0]
    if n_pts == 0:
        raise ValueError(f"No points loaded from {path}.")
    if n_pts != target_n:
        replace = n_pts < target_n
        idxs = np.random.choice(n_pts, target_n, replace=replace)
        pts = pts[idxs]

    # optional z-up correction happens later in pipeline using rotate_around_axis
    return pts

def main():
    checkpoints_folder = "checkpoints/normalized"  # specify your checkpoints folder
    checkpoint_file = "ckpt.pt"  # specify your checkpoint file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # <-- change input path here to your OBJ (or PLY) -->
    # 006_mustard_bottle
    path_to_input = "assets/ycb_inertia/035_power_drill/textured.obj"  # can be .obj, .ply, .stl, etc.
    z_up = False  # specify if your input point cloud/mesh is in z-up orientation
    normalize = True  # whether to normalize input
    lm_optimization = False  # use LM optimization
    resolution = 30  # resolution for mesh extraction
    target_n_points = 4096  # points sampled for SuperDec input

    ckp_path = os.path.join(checkpoints_folder, checkpoint_file)
    config_path = os.path.join(checkpoints_folder, 'config.yaml')
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)

    model = SuperDec(configs.superdec).to(device)
    model.lm_optimization = lm_optimization

    print("Loading checkpoint from:", ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # --- Load and sample (OBJ/mesh or PLY point cloud) ---
    points_np = load_and_sample_pointcloud(path_to_input, target_n=target_n_points, z_up=z_up, mesh_sample_method="uniform")

    # Normalize if requested
    if normalize:
        points, translation, scale = normalize_points(points_np)
    else:
        points = points_np.copy()
        translation = np.zeros(3)
        scale = 1.0

    if z_up:
        points = rotate_around_axis(points, axis=(1,0,0), angle=-np.pi/2, center_point=np.zeros(3))

    # To tensor
    points_t = torch.from_numpy(points).unsqueeze(0).to(device).float()

    with torch.no_grad():
        outdict = model(points_t)
        # move tensors to cpu for postproc
        for key in outdict:
            if isinstance(outdict[key], torch.Tensor):
                outdict[key] = outdict[key].cpu()
        translation = np.array([translation])
        scale = np.array([scale])
        outdict = denormalize_outdict(outdict, translation, scale, z_up)
        points_denorm = denormalize_points(points_t.cpu(), translation, scale, z_up)

    pred_handler = PredictionHandler.from_outdict(outdict, points_denorm, ['object'])
    mesh = pred_handler.get_meshes(resolution=resolution)[0]
    pcs = pred_handler.get_segmented_pcs()[0]

    # visualizer
    server = viser.ViserServer()
    server.scene.add_mesh_trimesh("superquadrics", mesh=mesh, visible=True)

    server.scene.add_point_cloud(
        name="/segmented_pointcloud",
        points=np.array(pcs.points),
        colors=np.array(pcs.colors),
        point_size=0.001,
    )
    if z_up:
        server.scene.set_up_direction([0.0, 0.0, 1.0])
    else:
        server.scene.set_up_direction([0.0, 1.0, 0.0])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.8, 0.8, 0.8)
        client.camera.look_at = (0., 0., 0.)
    while True:
        time.sleep(10.0)

if __name__ == "__main__":
    main()
