import numpy as np
import time
from src.dataset_objects import DatasetObjects
from src.sq_handler import SQHandler
from utils.utils_vis import visualize_with_viser


if __name__ == "__main__":
    ds = DatasetObjects("assets/ycb_datasets")

    obj_name = "035_power_drill"
    obj_info = ds.get_info(obj_name)
    # print(obj_info)

    # inertia, manifold, coacd, simplified
    mesh_base = ds.get_mesh(obj_name, "inertia")
    mesh_manifold = ds.get_mesh(obj_name, "manifold")
    mesh_coacd = ds.get_mesh(obj_name, "coacd")
    mesh_simplified = ds.get_mesh(obj_name, "simplified")
    convex_pieces = ds.get_mesh(obj_name, "convex_pieces")

    meshes_for_vis = {
        "mesh_base": mesh_base,
        "mesh_manifold": mesh_manifold,
        "mesh_coacd": mesh_coacd,
        "mesh_simplified": mesh_simplified,
        "convex_pieces": convex_pieces,
    }

    pcs_sample_poisson, _ = ds.get_point_cloud(
        obj_name, n_points=1024, method="poisson"
    )

    pointclouds_for_vis = {
        "pcs_sample_poisson": (pcs_sample_poisson, None),
    }

    visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
