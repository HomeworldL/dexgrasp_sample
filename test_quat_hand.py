import numpy as np
from scipy.spatial.transform import Rotation as R

def main():
    # 创建一个任意的初始旋转矩阵（这里用单位矩阵作为例子）
    initial_rotation = np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    print("初始旋转矩阵:")
    print(initial_rotation)

    # 或者或者创建一个特定的旋转矩阵作为示例
    # 例如：绕X轴旋转45度
    ry = R.from_euler('y', 45, degrees=True).as_matrix()
    
    
    # 矩阵乘法：初始旋转 * RY(30)
    result_rotation = initial_rotation @ ry
    print("相乘后的旋转矩阵:")
    print(result_rotation)
    
    # 转换为四元数
    result_quat = R.from_matrix(result_rotation).as_quat()
    print("结果的四元数表示 (Scipy格式 - [x, y, z, w]):")
    print(result_quat)
    
    # 如果想要 w 在前的格式
    result_quat_w_first = np.roll(result_quat, shift=1)
    print("结果的四元数表示 (w在前格式 - [w, x, y, z]):")
    print(result_quat_w_first)

if __name__ == "__main__":
    main()