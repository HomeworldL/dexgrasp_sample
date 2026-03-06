import numpy as np
from src.dataset_objects import DatasetObjects
from src.sq_handler import SQHandler
from utils.utils_vis import visualize_with_viser


def ndarray_to_vis_colors(cols):
    """
    Convert colors array to format acceptable by visualize_with_viser:
    Accepts float [0,1] or uint8 [0,255]. Returns either float or uint8 (both supported).
    We'll return uint8 when input is float for nicer display.
    """
    cols = np.asarray(cols)
    if cols.size == 0:
        return None
    if cols.dtype == np.float32 or cols.dtype == np.float64:
        # assume in [0,1]
        cols_u8 = np.clip((cols * 255.0).round(), 0, 255).astype(np.uint8)
        return cols_u8
    else:
        return cols.astype(np.uint8)


if __name__ == "__main__":
    ds = DatasetObjects("assets/ycb_datasets")

    obj_name = "035_power_drill"

    # inertia, manifold, coacd, simplified
    mesh_base = ds.get_mesh(obj_name, "inertia")
    mesh_manifold = ds.get_mesh(obj_name, "manifold")
    mesh_coacd = ds.get_mesh(obj_name, "coacd")
    mesh_simplified = ds.get_mesh(obj_name, "simplified")
    convex_pieces = ds.get_mesh(obj_name, "convex_pieces")

    superdec_data = ds.load_superdec_npz(obj_name)
    handler = SQHandler(superdec_data)
    # print(f"data: {superdec_data}")
    mesh_superdec = handler.get_mesh(0, resolution=64, colors=True)

    meshes_for_vis = {
        "mesh_base": mesh_base,
        "mesh_manifold": mesh_manifold,
        "mesh_coacd": mesh_coacd,
        "mesh_simplified": mesh_simplified,
        "convex_pieces": convex_pieces,
        "superdec_mesh": mesh_superdec,
    }

    pointclouds_for_vis = {}
    if superdec_data is not None:
        pcd = superdec_data.get("pcd", None)
        seg = superdec_data.get("segmentation", None)
        assign = superdec_data.get("assign_matrix", None)
        if pcd is not None:
            pts = np.asarray(pcd)
            # determine colors
            if seg is not None:
                seg = np.asarray(seg).astype(int)
                # generate palette
                num_labels = int(seg.max() + 1) if seg.size > 0 else 1
                palette = (
                    np.random.RandomState(0).randint(0, 255, size=(num_labels, 3))
                ).astype(np.uint8)
                cols = palette[seg % num_labels]
            elif assign is not None:
                assign = np.asarray(assign)
                if assign.ndim == 2:
                    labels = np.argmax(assign, axis=1).astype(int)
                elif assign.ndim == 3:
                    labels = np.argmax(assign[0], axis=1).astype(int)
                else:
                    labels = np.zeros((pts.shape[0],), dtype=int)
                num_labels = int(labels.max() + 1) if labels.size > 0 else 1
                palette = (
                    np.random.RandomState(1).randint(0, 255, size=(num_labels, 3))
                ).astype(np.uint8)
                cols = palette[labels % num_labels]
            else:
                # fallback white
                cols = np.ones((pts.shape[0], 3), dtype=np.uint8) * 220

            pointclouds_for_vis["superdec_pcd"] = {
                "points": pts,
                "colors": ndarray_to_vis_colors(cols),
                "point_size": 0.002,
            }

            # also add segmented pieces as separate small clouds (optional)
            if seg is not None:
                unique_labels = np.unique(seg)
                for lab in unique_labels:
                    mask = seg == lab
                    if np.count_nonzero(mask) == 0:
                        continue
                    pts_sub = pts[mask]
                    cols_sub = (
                        np.tile(
                            np.random.RandomState(int(lab) + 10).randint(
                                0, 255, size=(3,)
                            ),
                            (pts_sub.shape[0], 1),
                        )
                    ).astype(np.uint8)
                    pointclouds_for_vis[f"seg_{lab}"] = {
                        "points": pts_sub,
                        "colors": ndarray_to_vis_colors(cols_sub),
                        "point_size": 0.004,
                    }

    visualize_with_viser(meshes_for_vis, pointclouds_for_vis)
