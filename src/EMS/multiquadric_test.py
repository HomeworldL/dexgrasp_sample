import numpy as np
from EMS_recovery import EMS_recovery
from utilities import read_ply, showPoints
# from mayavi import mlab
from sklearn.cluster import DBSCAN

def hierarchical_ems(
    point,
    OutlierRatio=0.9,           # prior outlier probability [0, 1) (default: 0.1)
    MaxIterationEM=20,           # maximum number of EM iterations (default: 20)
    ToleranceEM=1e-3,            # absolute tolerance of EM (default: 1e-3)
    RelativeToleranceEM=2e-1,    # relative tolerance of EM (default: 1e-1)
    MaxOptiIterations=2,         # maximum number of optimization iterations per M (default: 2)
    Sigma=0.3,                   # initial sigma^2 (default: 0 - auto generate)
    MaxiSwitch=2,                # maximum number of switches allowed (default: 2)
    AdaptiveUpperBound=True,    # Introduce adaptive upper bound to restrict the volume of SQ (default: false)
    Rescale=False,                # normalize the input point cloud (default: true)
    MaxLayer=5,                  # maximum depth
    Eps=1.7,                    # IMPORTANT: varies based on the size of the input pointcoud (DBScan parameter)
    MinPoints=60,               # DBScan parameter required minimum points
):

    point_seg = {key: [] for key in list(range(0, MaxLayer+1))}
    point_outlier = {key: [] for key in list(range(0, MaxLayer+1))}
    point_seg[0] = [point]
    list_quadrics = []
    quadric_count = 1
    for h in range(MaxLayer):
        for c in range(len(point_seg[h])):
            print(f"Counting number of generated quadrics: {quadric_count}")
            quadric_count += 1
            x_raw, p_raw = EMS_recovery(
                point_seg[h][c],
                OutlierRatio,
                MaxIterationEM,
                ToleranceEM,
                RelativeToleranceEM,
                MaxOptiIterations,
                Sigma,
                MaxiSwitch,
                AdaptiveUpperBound,
                Rescale,
            )
            point_previous = point_seg[h][c]
            list_quadrics.append(x_raw)
            outlier = point_seg[h][c][p_raw < 0.1, :]
            point_seg[h][c] = point_seg[h][c][p_raw > 0.1, :]
            if np.sum(p_raw) < (0.8 * len(point_previous)):

                print("DEBUG outlier type:", type(outlier))
                try:
                    print(" DEBUG shape:", getattr(outlier, "shape", None))
                except Exception:
                    pass
                outarr = np.asarray(outlier)
                print(" DEBUG np.asarray -> dtype:", outarr.dtype, "ndim:", outarr.ndim, "shape:", outarr.shape)
                print(" DEBUG flags:", outarr.flags if hasattr(outarr, "flags") else None)
                # show first few elements types if object dtype
                if outarr.dtype == object:
                    print(" DEBUG first elements types:", [type(x) for x in outarr[:5]])
                # finite check
                try:
                    print(" DEBUG finite counts:", np.isfinite(outarr).all(axis=1).sum(), " / ", outarr.shape[0])
                except Exception:
                    print(" DEBUG cannot check finite (probably ragged/object array)")

                clustering = DBSCAN(eps=Eps, min_samples=MinPoints).fit(outlier)
                labels = list(set(clustering.labels_))
                labels = [item for item in labels if item >= 0]
                if len(labels) >= 1:
                    for i in range(len(labels)):
                        point_seg[h + 1].append(outlier[clustering.labels_ == i])
                point_outlier[h].append(outlier[clustering.labels_ == -1])
            else:
                point_outlier[h].append(outlier)
    return point_seg, point_outlier, list_quadrics


# Load pointcloud 
point_cloud = read_ply("/home/ccs/repositories/sq_dex_grasp/inertia_watertight.ply")
point_seg, point_outlier, list_quadrics = hierarchical_ems(point_cloud)

# -----------    Plot multiquadric figure --------------
# fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
# for quadric in list_quadrics:
#     quadric.showSuperquadric(arclength=0.2)
# showPoints(point_cloud, scale_factor=0.001)
# mlab.show()
