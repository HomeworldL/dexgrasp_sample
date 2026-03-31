import numpy as np
import math

# 机械手参数
unit_length = 1000
module1_config = {
    "p1": np.array([5.355 / unit_length, -6.0 / unit_length, 17.185 / unit_length]),
    "p2": np.array([5.355 / unit_length, 6.0 / unit_length, 17.185 / unit_length]),
    "p3": np.array([14.050 / unit_length, -6.0 / unit_length, 0]),
    "p4": np.array([14.050 / unit_length, 6.0 / unit_length, 0]),
    "l1": 22.9 / unit_length,
    "l2": 22.9 / unit_length,
    "z0": 4.0 / unit_length,
    "bevel": 256 * 64 * 2,
    # "screw": 2 * math.pi / 0.001,
    "h": 0.001,
    "screw": 1 * 256 * 4 / 0.001,
    "A": 9.5 / unit_length,
    "B": 32.35 / unit_length,
    "C": 29.867 / unit_length,
    "D": 7.3 / unit_length,
    "degree1_init": math.radians(61.4845),
    "degree2_init": math.radians(118.1266),
}
module2_config = {
    "p1": np.array([9.932 / unit_length, 0.0 / unit_length, 13.797 / unit_length]),
    "p2": np.array([0.0 / unit_length, 1.0 / unit_length, 0.0 / unit_length]),
    "p3": np.array([13.050 / unit_length, 0.0 / unit_length, 0]),
    "l1": 5.6 / unit_length,
    "l2": 18.4 / unit_length,
    "z0": 4.0 / unit_length,
    "bevel": 128 * 256 * 2,
    "screw": 4 * 256 * 4 / 0.001,
    "h": 0.001,
    "A": 4.5 / unit_length,
    "B": 26.5 / unit_length,
    "C": 25.0 / unit_length,
    "D": 6.0 / unit_length,
    "degree1_init": math.radians(45.0619),
    "degree2_init": math.radians(124.5317),
    # mcp-pip
    "Am": 7.0 / unit_length,
    "Bm": 46.0 / unit_length,
    "Cm": 45.0 / unit_length,
    "Dm": 6.0 / unit_length,
    "degree1m_init": math.radians(66.3557),
    "degree2m_init": math.radians(111.6954),
}
module3_config = {
    "p1": np.array([9.932 / unit_length, 0.0 / unit_length, 13.797 / unit_length]),
    "p2": np.array([0.0 / unit_length, 1.0 / unit_length, 0.0 / unit_length]),
    "p3": np.array([13.050 / unit_length, 0.0 / unit_length, 0]),
    "l1": 5.6 / unit_length,
    "l2": 18.4 / unit_length,
    "z0": 4.0 / unit_length,
    "bevel": 128 * 256 * 2,
    "screw": 4 * 256 * 4 / 0.001,
    "h": 0.001,
    "A": 4.5 / unit_length,
    "B": 26.5 / unit_length,
    "C": 25.0 / unit_length,
    "D": 6.0 / unit_length,
    "degree1_init": math.radians(45.0619),
    "degree2_init": math.radians(124.5317),
    # mcp-pip
    "Am": 7.0 / unit_length,
    "Bm": 36.85 / unit_length,
    "Cm": 36.0 / unit_length,
    "Dm": 6.0 / unit_length,
    "degree1m_init": math.radians(67.63534),
    "degree2m_init": math.radians(103.46481),
}
module4_config = {
    # "screw": 1 * 256 * 4 / 0.001,
    "h": 0.001,
    "A": 7.0 / unit_length,
    "B": 37.61 / unit_length,
    "C": 36.0 / unit_length,
    "D": 5.5 / unit_length,
    "degree1_init": math.radians(70.96005765),  # 上一版71.98912
    "degree2_init": math.radians(111.50535364),
    "A1": 7.10828714 / unit_length,  # 横向距离
    "B1": 8.5 / unit_length,  # 纵向距离
    "degree1_init1": math.radians(50.09530324),  # 水平夹角
    "A2": 7.20179328 / unit_length,
    "B2": 9 / unit_length,
    "degree1_init2": math.radians(51.3332312),
}
module_config = {
    "thumb": module4_config,
    "index": module1_config,
    "middle": module1_config,
    "ring": module2_config,
    "little": module3_config,
}
finger_name = ["index", "middle", "ring", "little", "thumb"]

import matplotlib.pyplot as plt

np.set_printoptions(precision=6, suppress=True, floatmode="fixed")
finger_joint_id = {
    "index": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "ring": [8, 9, 10, 11],
    "little": [12, 13, 14, 15],
    "thumb": [16, 17, 18, 19],
}

joint_limit_low = [
    -0.4,
    0.0,
    0.0,
    0.0,
    -0.4,
    0.0,
    0.0,
    0.0,
    -0.4,
    0.0,
    0.0,
    0.0,
    -0.4,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
joint_limit_up = [
    0.4,
    2.09,
    1.744,
    1.744,
    0.4,
    2.09,
    1.744,
    1.744,
    0.4,
    1.744,
    1.744,
    1.744,
    0.4,
    1.6,
    1.744,
    1.744,
    1.9,
    0.8,
    1.744,
    1.744,
]


# %% 用于计算四杆机构
def get_module_config(name: str):
    return module_config[name]


def four_bar_forward(q, A, B, C, D, degree1_init, degree2_init):
    """求解四杆机构角度关系,例如从PIP推导DIP"""
    L1 = math.sqrt(A**2 + C**2 - 2 * A * C * math.cos(q + degree1_init))
    parameter1 = (C - A * math.cos(q + degree1_init)) / L1
    parameter2 = (L1**2 + D**2 - B**2) / (2 * L1 * D)
    parameter1 = max(min(parameter1, 1), -1)
    parameter2 = max(min(parameter2, 1), -1)
    return degree2_init - (math.acos(parameter2) - math.acos(parameter1))


def four_bar_backward(q, A, B, C, D, degree1_init, degree2_init):
    L2 = math.sqrt(C**2 + D**2 - 2 * C * D * math.cos(degree2_init - q))
    parameter1 = math.sin(degree2_init - q) * D / L2
    parameter2 = (L2 ^ 2 + A ^ 2 - B ^ 2) / (2 * L2 * A)
    parameter1 = max(min(parameter1, 1), -1)
    parameter2 = max(min(parameter2, 1), -1)
    return math.acos(parameter2) - math.asin(parameter1) - degree1_init


def four_bar_pip2dip(q: float, name: str):
    """所有手指的dip都与pip耦合

    :param q: _description_
    :type q: float
    :param name: _description_
    :type name: str
    :return: _description_
    :rtype: _type_
    """
    cfg = get_module_config(name)
    return four_bar_forward(
        q,
        cfg["A"],
        cfg["B"],
        cfg["C"],
        cfg["D"],
        cfg["degree1_init"],
        cfg["degree2_init"],
    )


def four_bar_dip2pip(q: float, name: str):
    cfg = get_module_config(name)
    return four_bar_backward(
        q,
        cfg["A"],
        cfg["B"],
        cfg["C"],
        cfg["D"],
        cfg["degree1_init"],
        cfg["degree2_init"],
    )


def four_bar_mcp2pip(q: float, name: str):
    assert name == "ring" or name == "little"
    cfg = get_module_config(name)
    return four_bar_forward(
        q,
        cfg["Am"],
        cfg["Bm"],
        cfg["Cm"],
        cfg["Dm"],
        cfg["degree1m_init"],
        cfg["degree2m_init"],
    )


def four_bar_pip2mcp(q: float, name: str):
    assert name == "ring" or name == "little"
    cfg = get_module_config(name)
    return four_bar_backward(
        q,
        cfg["Am"],
        cfg["Bm"],
        cfg["Cm"],
        cfg["Dm"],
        cfg["degree1m_init"],
        cfg["degree2m_init"],
    )


polys = {}


def four_bar_fit_polynomial(degree=4, num_points=1000):
    # 拟合关节参数

    print(joint_limit_low)
    print(joint_limit_up)
    finger_names = ["index", "middle", "ring", "little", "thumb"]
    for name in finger_names:
        if name in ["ring", "little"]:
            joint_ids = [
                finger_joint_id[name][1],
                finger_joint_id[name][2],
            ]
            funcs = [four_bar_mcp2pip, four_bar_pip2dip]
            poly_keys = [f"{name}_pip", f"{name}_dip"]
        else:
            joint_ids = [finger_joint_id[name][2]]
            funcs = [four_bar_pip2dip]
            poly_keys = [f"{name}_dip"]

        for joint_id, func, poly_key in zip(joint_ids, funcs, poly_keys):
            qs = np.linspace(
                joint_limit_low[joint_id],
                joint_limit_up[joint_id],
                num_points,
            )
            print(joint_limit_low[joint_id])
            print(joint_limit_up[joint_id])
            outputs = [func(q, name) for q in qs]
            # print(qs)
            # print(outputs)
            coeffs = np.polyfit(qs, outputs, degree)
            polys[poly_key] = np.poly1d(coeffs)

            coeffs_reversed = coeffs[::-1]
            print(coeffs_reversed)

            # print(f"q = qs[0] {qs[0]}: output = {polys[poly_key](qs[0])}")
            # print(f"q = 1: output = {polys[poly_key](1)}")
            # print(f"q = qs[-1] {qs[-1]}: output = {polys[poly_key](qs[-1])}")

            # 可视化拟合结果
            plt.plot(qs, outputs, "o", label="Original Data")
            plt.plot(qs, polys[poly_key](qs), "-", label="Fitted Polynomial")
            plt.legend()
            plt.xlabel("q")
            plt.ylabel("Output")
            plt.title(f"Four Bar Mechanism {poly_key} Fitting")
            plt.show()

    # print(type(qs))
    # print(type(outputs))
    # print(type(poly(qs)))
    # print(type(poly(qs[0])))
    # print(poly(qs[0]))


four_bar_fit_polynomial(4, 1000)

