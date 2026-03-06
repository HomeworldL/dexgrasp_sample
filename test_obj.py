import numpy as np
import time
from src.dataset_objects import DatasetObjects
from src.sq_handler import SQHandler
from utils.utils_vis import visualize_with_viser


if __name__ == "__main__":
    ds = DatasetObjects("assets/ycb_datasets")

    # obj_name = "002_master_chef_can" # 002_master_chef_can 035_power_drill
    obj_name = ds.id2name[28]
    obj_name = "013_apple"

    obj_info = ds.get_info(obj_name)
    # print(obj_info)

    # inertia, manifold, coacd, simplified
    mesh_base = ds.get_mesh(obj_name, "inertia")
    mesh_manifold = ds.get_mesh(obj_name, "manifold")
    mesh_coacd = ds.get_mesh(obj_name, "coacd")
    mesh_simplified = ds.get_mesh(obj_name, "simplified")
    convex_pieces = ds.get_mesh(obj_name, "convex_pieces")

    points_dict, ems_params_dict, net_params_dict = ds.load_superdec_npz(obj_name)
    handler = SQHandler(points_dict, ems_params_dict, obj_name)
    handler.print_info()
    # print(f"data: {superdec_data}")
    mesh_superdec = handler.get_meshes(resolution=64, colors=True)
    # mesh_superdec = handler.get_mesh(0, resolution=64, colors=True)

    meshes_for_vis = {
        "mesh_base": mesh_base,
        # "mesh_manifold": mesh_manifold,
        # "mesh_coacd": mesh_coacd,
        # "mesh_simplified": mesh_simplified,
        # "convex_pieces": convex_pieces,
        "superdec_mesh": mesh_superdec,
    }

    pcs_sample_poisson, _ = ds.get_point_cloud(
        obj_name, n_points=1024, method="poisson"
    )
    pcs_sample_uniform, _ = ds.get_point_cloud(
        obj_name, n_points=1024, method="uniform"
    )

    pcs_segmented, cols_segmented = handler.get_segmented_pcs()
    # pcs_segmented, cols_segmented = handler.get_segmented_pc(0)
    # print(pcs_segmented.shape, cols_segmented.shape)

    ts = time.time()
    pcs_sample_uv, normals_sample_uv = handler.sample_surface_uv(
        0, Nu=100, Nv=100, d=0.0
    )
    print(f"sample_uv: {time.time() - ts}")
    ts = time.time()
    pcs_sample, normals_sample = handler.sample_surface(0)
    print(f"pcs_sample: {pcs_sample.shape}")
    print(f"sample: {time.time() - ts}")

    pointclouds_for_vis = {
        "pcs_segmented": (pcs_segmented, cols_segmented),
        "pcs_sample_uv": (pcs_sample_uv, None),
        "pcs_sample": (pcs_sample, None),
        "pcs_sample_poisson": (pcs_sample_poisson, None),
        # "pcs_sample_uniform": (pcs_sample_uniform, None),
    }

    visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
