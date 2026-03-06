#!/usr/bin/env python3
import sys
import os

sys.path.append(os.path.realpath("."))

import numpy as np
from pathlib import Path

from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from dm_control import mjcf
import mink
from loop_rate_limiters import RateLimiter
import time
import threading
import cv2

HOME_QPOS = []
HOME_QPOS.extend([0, 0, 0, 1, 0, 0, 0])
HOME_QPOS.extend([0, 0.5, 0.0, 0.0])
HOME_QPOS.extend([-0.1, 0.5, 0.3, 0.3])
HOME_QPOS.extend([-0.25, 0.55, 0.55, 0.55])
HOME_QPOS.extend([-0.4, 0.6, 0.6, 0.6])
HOME_QPOS.extend([0.3, 0.0, 0.1, 0.1])

HOME_CTRL = []
HOME_CTRL.extend([0, 0.5, 0.0])
HOME_CTRL.extend([-0.1, 0.5, 0.3])
HOME_CTRL.extend([-0.25, 0.55])
HOME_CTRL.extend([-0.4, 0.6])
HOME_CTRL.extend([0.3, 0.0, 0.1])


class CameraTracker:
    def __init__(self):
        self.camera = mujoco.MjvCamera()

    def sync_from_viewer(self, viewer_cam):
        """从viewer同步相机参数"""
        self.camera.type = viewer_cam.type
        self.camera.fixedcamid = viewer_cam.fixedcamid
        self.camera.trackbodyid = viewer_cam.trackbodyid
        self.camera.lookat[:] = viewer_cam.lookat
        self.camera.distance = viewer_cam.distance
        self.camera.azimuth = viewer_cam.azimuth
        self.camera.elevation = viewer_cam.elevation


np.set_printoptions(precision=6, suppress=True, floatmode="fixed")

_HERE = Path(__file__).parent


def construct_model(robot_xml, home_qpos=None, home_ctrl=None):
    robot_mjcf = mjcf.from_path(robot_xml.as_posix())
    try:
        robot_mjcf.find("key", "home").remove()
    except:
        pass
    robot_mjcf.keyframe.add("key", name="home", qpos=home_qpos, ctrl=home_ctrl)

    return robot_mjcf


# ----------------------
# 主流程
# ----------------------
def main():

    robot_xml = _HERE / "assets" / "hands" / "liberhand" / f"liberhand_right.xml"

    mjcf_model = construct_model(robot_xml, home_qpos=HOME_QPOS, home_ctrl=HOME_CTRL)
    # print(mjcf_model.to_xml_string())
    model = mujoco.MjModel.from_xml_string(
        mjcf_model.to_xml_string(), mjcf_model.get_assets()
    )
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    renderer = mujoco.Renderer(model, height=2160, width=3840)
    tracker = CameraTracker()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            # 同步相机参数
            tracker.sync_from_viewer(viewer.cam)

            # 使用同步后的相机进行渲染
            renderer.update_scene(data, camera=tracker.camera)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("test", bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # 按q退出
                break
            elif key == ord("s"):  # 按s保存图片
                filename = f"./plot_outputs/paper/screenshot_hand.png"
                # MuJoCo返回的是RGB格式，但OpenCV保存需要BGR格式
                cv2.imwrite(filename, bgr)
                print(f"截图已保存为: {filename}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
