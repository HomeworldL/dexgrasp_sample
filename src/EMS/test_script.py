import sys
import os

sys.path.append(os.path.realpath("."))

# from mayavi import mlab
import argparse
import sys
import numpy as np

from src.EMS.utilities import read_ply, getSuperquadricsMesh
from src.EMS.EMS_recovery import EMS_recovery

import timeit

import viser
import time

def main(argv):

    parser = argparse.ArgumentParser(
        description='Probabilistic Recovery of a superquadric surface from a point cloud file *.ply.')

    parser.add_argument(
        'path_to_data',
        # default = '~/EMS-probabilistic_superquadric_fitting/MATLAB/example_scripts/data/noisy_pointCloud_example_1.ply',
        help='Path to the directory containing the point cloud file *.ply.'
    )

    parser.add_argument(
        '--visualize',
        action = 'store_true',
        help='Visualize the recoverd superquadric and the input point cloud.'
    )

    parser.add_argument(
        '--runtime',
        action = 'store_true',
        help='Show the runtime.'
    )

    parser.add_argument(
        '--result',
        action = 'store_true',       
        help='Print the recovered superquadric parameter.'
    )

    parser.add_argument(
        '--outlierRatio',
        type = float,
        default = 0.2,       
        help='Set the prior outlier ratio. Default is 0.2.'
    )

    parser.add_argument(
        '--adaptiveUpperBound',
        action = 'store_true',       
        help='Implemet addaptive upper bound to limit the volume of the superquadric.'
    )

    parser.add_argument(
        '--arcLength',
        type = float,
        default = 0.2,       
        help='Set the arclength (resolution) for rendering the superquadric. Default is 0.2.'
    )

    parser.add_argument(
        '--pointSize',
        type = float,
        default = 0.1,       
        help='Set the point size for plotting the point cloud. Default is 0.2.'
    )

    args = parser.parse_args(argv)
    
    print('----------------------------------------------------')
    print('Loading point cloud from: ', args.path_to_data, '...')
    point = read_ply(args.path_to_data)
    print(f"shape of point cloud: {point.shape}")
    print('Point cloud loaded.')
    print('----------------------------------------------------')

    # first run to eliminate jit compiling time
    sq_recovered, p = EMS_recovery(point)

    start = timeit.default_timer()
    sq_recovered, p = EMS_recovery(point, 
                                   OutlierRatio=args.outlierRatio, 
                                   AdaptiveUpperBound=args.adaptiveUpperBound
                      )
    stop = timeit.default_timer()
    print('Superquadric Recovered.')
    if args.runtime is True:
        print('Runtime: ', (stop - start) * 1000, 'ms')
    print('----------------------------------------------------')
    
    if args.result is True:
        print('shape =', sq_recovered.shape)
        print('scale =', sq_recovered.scale)
        print('euler =', sq_recovered.euler)
        print('translation =', sq_recovered.translation)
        print('----------------------------------------------------')
    
    if args.visualize is True:
        # 启动 viser server（在创建时会打开本地 webserver）
        server = viser.ViserServer()
        scene = server.scene

        # 1) 得到 superquadric 网格
        verts, faces = getSuperquadricsMesh(sq_recovered, arclength=args.arcLength)

        # 2) 将网格加入 scene（color 使用 0-255 RGB）
        # name 可以是 "/sq" 或自定义路径；position 与 wxyz 可选
        scene.add_mesh_simple(name="/superquadric", vertices=verts, faces=faces,
                            color=(0, 0, 255), opacity=0.8)

        # 3) 将点云加入 scene
        pts = np.asarray(point, dtype=np.float32)
        # scene API will cast precision internally; colors can be a single RGB tuple
        scene.add_point_cloud(name="/pointcloud", points=pts, colors=(255, 0, 0), point_size=0.01)


        print("Open your browser to http://localhost:8080 to view the scene. Press Ctrl+C to exit.")
        try:
            # server runs in background thread; keep main thread alive
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting visualization.")
            server.close()  # 如果 API 支持关闭（若没有，删掉此行）


if __name__ == "__main__":
    main(sys.argv[1:])