# from mayavi import mlab
import numpy as np
import plyfile

def showSuperquadrics(x, threshold = 1e-2, num_limit = 10000, arclength = 0.02):
    # avoid numerical instability in sampling
    if x.shape[0] < 0.007:
        x.shape[0] = 0.007
    if x.shape[1] < 0.007:
        x.shape[1] = 0.007
    # sampling points in superellipse    
    point_eta = uniformSampledSuperellipse(x.shape[0], [1, x.scale[2]], threshold, num_limit, arclength)
    point_omega = uniformSampledSuperellipse(x.shape[1], [x.scale[0], x.scale[1]], threshold, num_limit, arclength)
    
    # preallocate meshgrid
    x_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
    y_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))
    z_mesh = np.ones((np.shape(point_omega)[1], np.shape(point_eta)[1]))

    for m in range(np.shape(point_omega)[1]):
        for n in range(np.shape(point_eta)[1]):
            point_temp = np.zeros(3)
            point_temp[0 : 2] = point_omega[:, m] * point_eta[0, n]
            point_temp[2] = point_eta[1, n]
            point_temp = x.RotM @ point_temp + x.translation

            x_mesh[m, n] = point_temp[0]
            y_mesh[m, n] = point_temp[1]
            z_mesh[m, n] = point_temp[2]
    
    # mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    # mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0, 0, 1), opacity=0.8)

def getSuperquadricsMesh(x, threshold=1e-2, num_limit=10000, arclength=0.02):
    """
    从 superquadric 类对象 x（具有 .shape(2,), .scale(3,), .RotM (3x3), .translation (3,)）
    采样得到三角网格 (vertices, faces).
    返回：
      vertices: np.ndarray (V, 3), dtype float32
      faces:    np.ndarray (F, 3), dtype int32
    """
    # safeguard small values (与你之前代码逻辑一致)
    eps1 = float(x.shape[0]) if np.isscalar(x.shape[0]) else float(x.shape[0])
    eps2 = float(x.shape[1]) if np.isscalar(x.shape[1]) else float(x.shape[1])
    # use provided sampling function to get 1/4 + mirrored samples
    point_eta = uniformSampledSuperellipse(x.shape[0], [1.0, x.scale[2]], threshold, num_limit, arclength)
    point_omega = uniformSampledSuperellipse(x.shape[1], [x.scale[0], x.scale[1]], threshold, num_limit, arclength)

    No = np.shape(point_omega)[1]   # number of samples along omega
    Ne = np.shape(point_eta)[1]     # number of samples along eta

    # build vertices array: (No * Ne, 3)
    verts = np.zeros((No * Ne, 3), dtype=np.float32)

    idx = 0
    for m in range(No):
        for n in range(Ne):
            ptemp = np.zeros(3, dtype=float)
            ptemp[0:2] = point_omega[:, m] * point_eta[0, n]
            ptemp[2] = point_eta[1, n]
            # transform to global coordinates using RotM and translation
            pglob = x.RotM @ ptemp + x.translation
            verts[idx, :] = pglob.astype(np.float32)
            idx += 1

    # build faces (two triangles per quad cell)
    faces = []
    # grid indexing: row m (0..No-1), col n (0..Ne-1)
    # linear index: id = m * Ne + n
    for m in range(No - 1):
        for n in range(Ne - 1):
            v0 = m * Ne + n
            v1 = v0 + 1
            v2 = v0 + Ne
            v3 = v2 + 1
            # two triangles (v0, v1, v2) and (v1, v3, v2)
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.asarray(faces, dtype=np.int32)

    return verts, faces

def uniformSampledSuperellipse(epsilon, scale, threshold = 1e-2, num_limit = 10000, arclength = 0.02):

    # initialize array storing sampled theta
    theta = np.zeros(num_limit)
    theta[0] = 0

    for i in range(num_limit):
        dt = dtheta(theta[i], arclength, threshold, scale, epsilon)
        theta_temp = theta[i] + dt

        if theta_temp > np.pi / 4:
            theta[i + 1] = np.pi / 4
            break
        else:
            if i + 1 < num_limit:
                theta[i + 1] = theta_temp
            else:
                raise Exception(
                'Number of the sampled points exceed the preset limit', \
                num_limit,
                'Please decrease the sampling arclength.'
                )
    critical = i + 1

    for j in range(critical + 1, num_limit):
        dt = dtheta(theta[j], arclength, threshold, np.flip(scale), epsilon)
        theta_temp = theta[j] + dt
        
        if theta_temp > np.pi / 4:
            break
        else:
            if j + 1 < num_limit:
                theta[j + 1] = theta_temp
            else:
                raise Exception(
                'Number of the sampled points exceed the preset limit', \
                num_limit,
                'Please decrease the sampling arclength.'
                )
    num_pt = j
    theta = theta[0 : num_pt + 1]

    point_fw = angle2points(theta[0 : critical + 1], scale, epsilon)
    point_bw = np.flip(angle2points(theta[critical + 1: num_pt + 1], np.flip(scale), epsilon), (0, 1))
    point = np.concatenate((point_fw, point_bw), 1)
    point = np.concatenate((point, np.flip(point[:, 0 : num_pt], 1) * np.array([[-1], [1]]), 
                           point[:, 1 : num_pt + 1] * np.array([[-1], [-1]]),
                           np.flip(point[:, 0 : num_pt], 1) * np.array([[1], [-1]])), 1)

    return point

def dtheta(theta, arclength, threshold, scale, epsilon):
    # calculation the sampling step size
    if theta < threshold:
        dt = np.abs(np.power(arclength / scale[1] +np.power(theta, epsilon), \
             (1 / epsilon)) - theta)
    else:
        dt = arclength / epsilon * ((np.cos(theta) ** 2 * np.sin(theta) ** 2) /
             (scale[0] ** 2 * np.cos(theta) ** (2 * epsilon) * np.sin(theta) ** 4 +
             scale[1] ** 2 * np.sin(theta) ** (2 * epsilon) * np.cos(theta) ** 4)) ** (1 / 2)
    
    return dt

def angle2points(theta, scale, epsilon):

    point = np.zeros((2, np.shape(theta)[0]))
    point[0] = scale[0] * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** epsilon
    point[1] = scale[1] * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** epsilon

    return point


def read_ply(path_to_file):
    # read points from a .ply file and store in an nparray
    plydata = plyfile.PlyData.read(path_to_file)
    pc = plydata['vertex'].data
    return np.array([[x, y, z] for x, y, z in pc])


def showPoints(point, scale_factor=0.1):
    pass
    
    # mlab.view(azimuth=0.0, elevation=0.0, distance=2)
    # mlab.points3d(point[:, 0], point[:, 1], point[:, 2], scale_factor=scale_factor, color=(1, 0, 0))