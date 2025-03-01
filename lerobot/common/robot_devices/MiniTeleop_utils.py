import sys
import os
from lerobot.common.utils.utils import (
    init_hydra_config,
    init_logging,
    log_say,
    none_or_int,
)
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
import time

from lerobot.common.robot_devices.utils import busy_wait
import pybullet as pb
import json
import numpy as np
import yaml

record_time_s = 30
fps = 60


# change directory to the root of the project
# os.chdir("/home/rhos/ziyu/so100/lerobot")


class LeadArmReader:
    URDF_DICT = {
        "koch": "urdf/assets/low_cost_robot_description/urdf/low_cost_robot.urdf",
        "so100": "urdf/assets/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf",
    }

    def __init__(self, robot_path, robot_overrides, visualize=True):
        self.robot_path = robot_path
        self.robot_overrides = robot_overrides

        self.robot = None
        self.arm = None

        if visualize:
            pb.connect(pb.GUI)
        else:
            pb.connect(pb.DIRECT)
        self.debug_lines = []
        self.init_arm()

    def init_arm(self):
        robot_cfg = init_hydra_config(self.robot_path, self.robot_overrides)
        self.robot_type = robot_cfg["robot_type"]
        urdf_path = self.URDF_DICT.get(self.robot_type, None)
        print(f'urdf: {urdf_path}')
        self.robot_pb = pb.loadURDF(
            urdf_path,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[
                0, 0, 1, 0] if self.robot_type == "so100" else [0, 0, 0, 1],
            useFixedBase=True,
        )

        self.follow_arm = pb.loadURDF(
            "urdf/assets/leap_hand/flexiv_leap.urdf",
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0.7071068, 0.7071068],
            useFixedBase=True,
        )

        self.robot = make_robot(robot_cfg)
        self.robot.connect()
        self.arm = self.robot.leader_arms["main"]

    def clear_debug_lines(self):
        for line in self.debug_lines:
            pb.removeUserDebugItem(line)
        self.debug_lines = []

    def get_lead_arm_eef(self):
        start_time = time.perf_counter()

        data = self.arm.read("Present_Position")

        if self.robot_type == "koch":
            # Convert angles to radians and adjust joint positions
            data = [
                -data[0],
                90 - data[1],
                90 - data[2],
                90 - data[3],
                data[4] + 90,
                -data[5],
            ]
            data = [angle * 3.14159265359 / 180 for angle in data]

            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, joint.item())

            # return the 6d pose of the end effector
            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = pb.getLinkState(self.robot_pb, 4)[1]

            dt_s = time.perf_counter() - start_time
            busy_wait(1 / fps - dt_s)

            return eef_pos, eef_orn, -data[-1]
        elif self.robot_type == "so100":

            data = [
                -data[0],
                90 - data[1],
                data[2] - 90,
                data[3] - 90,
                90 - data[4],
                data[5],
            ]
            data = [angle * 3.14159265359 / 180 for angle in data]

            for i, joint in enumerate(data):
                pb.resetJointState(self.robot_pb, i, joint.item())

            eef_pos = np.array(pb.getLinkState(self.robot_pb, 4)[0])
            eef_orn = pb.getLinkState(self.robot_pb, 4)[1]

            dt_s = time.perf_counter() - start_time
            busy_wait(1 / fps - dt_s)

            self.clear_debug_lines()
            eef_orn_matrix = np.array(
                pb.getMatrixFromQuaternion(eef_orn)).reshape(3, 3)
            original_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            z_in_eef = np.array([-1, 0, 0])
            x_in_eef = np.array([0, -1, 0])
            y_in_eef = np.array([0, 0, 1])
            new_axes = np.array([x_in_eef, y_in_eef, z_in_eef]).T
            # 添加一行1
            new_axes = np.concatenate(
                [new_axes, np.array([[1, 1, 1]])], axis=0)

            x_ab = np.array(
                [
                    np.dot(eef_orn_matrix[:, 0], original_axes[0]),
                    np.dot(eef_orn_matrix[:, 0], original_axes[1]),
                    np.dot(eef_orn_matrix[:, 0], original_axes[2]),
                ]
            )
            y_ab = np.array(
                [
                    np.dot(eef_orn_matrix[:, 1], original_axes[0]),
                    np.dot(eef_orn_matrix[:, 1], original_axes[1]),
                    np.dot(eef_orn_matrix[:, 1], original_axes[2]),
                ]
            )
            z_ab = np.array(
                [
                    np.dot(eef_orn_matrix[:, 2], original_axes[0]),
                    np.dot(eef_orn_matrix[:, 2], original_axes[1]),
                    np.dot(eef_orn_matrix[:, 2], original_axes[2]),
                ]
            )

            T_ab = np.array([x_ab, y_ab, z_ab, eef_pos]).T
            T_ab = np.concatenate([T_ab, np.array([[0, 0, 0, 1]])], axis=0)

            rot_axes = T_ab @ new_axes
            rot_axes = rot_axes[:3]
            # print(rot_axes)
            self.debug_lines.append(
                pb.addUserDebugLine(eef_pos, eef_pos + 0.1 *
                                    rot_axes[:, 0], [1, 0, 0], 5)
            )
            self.debug_lines.append(
                pb.addUserDebugLine(eef_pos, eef_pos + 0.1 *
                                    rot_axes[:, 1], [0, 1, 0], 5)
            )
            self.debug_lines.append(
                pb.addUserDebugLine(eef_pos, eef_pos + 0.1 *
                                    rot_axes[:, 2], [0, 0, 1], 5)
            )
            new_eef_orn = self.matrix2quaternion(rot_axes)
            print(new_eef_orn)

            return eef_pos, new_eef_orn, -data[-1]
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    def matrix2quaternion(self, matrix):
        tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (matrix[2, 1] - matrix[1, 2]) / S
            qy = (matrix[0, 2] - matrix[2, 0]) / S
            qz = (matrix[1, 0] - matrix[0, 1]) / S
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            S = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (matrix[0, 1] + matrix[1, 0]) / S
            qz = (matrix[0, 2] + matrix[2, 0]) / S
        elif matrix[1, 1] > matrix[2, 2]:
            S = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / S
            qx = (matrix[0, 1] + matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (matrix[1, 2] + matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / S
            qx = (matrix[0, 2] + matrix[2, 0]) / S
            qy = (matrix[1, 2] + matrix[2, 1]) / S
            qz = 0.25 * S
        return np.array([qx, qy, qz, qw])

    def get_follow_arm_joints(self):
        eef_pos, eef_orn, gripper = self.get_lead_arm_eef()

        eef_pos = eef_pos * 4  # soarm 2.5

        joints = pb.calculateInverseKinematics(
            self.follow_arm, 8, eef_pos, eef_orn)
        for i, joint in enumerate(joints):
            pb.resetJointState(self.follow_arm, i, joint)

        return joints, gripper

    def disconnect(self):
        self.arm.disconnect()

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])


if __name__ == "__main__":
    robot_path = "lerobot/configs/robot/koch.yaml"
    robot_overrides = ["~cameras", "~follower_arms"]
    reader = LeadArmReader(
        robot_path,
        robot_overrides,
    )

    while True:
        joints, gripper = reader.get_follow_arm_joints()

    reader.disconnect()
    pb.disconnect()
