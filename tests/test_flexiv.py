# Written by Yushun Xiang
import flexivrdk
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

# from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
# from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera
import time
from typing import List
from lerobot.common.robot_devices.utils import busy_wait
import pybullet as pb
import json
import torch
import numpy as np

# sys.path.append(os.path.join(os.path.dirname(__file__), "../tests"))
from tests.read_leader import LeadArmReader
from tests.camera import CameraD400

sys.path.append(os.path.join(os.path.dirname(__file__), "../lib_py"))

record_time_s = 30
fps = 60
frequency = 60  # Hz


# change directory to the root of the project
# os.chdir("/home/rhos/ziyu/so100/lerobot")
robot_ip = "192.168.2.100"
local_ip = "192.168.2.102"
log = flexivrdk.Log()
mode = flexivrdk.Mode


def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def self_exam(log):
    global robot
    try:
        robot = flexivrdk.Robot(robot_ip, local_ip)

        # Clear fault on robot server if any
        if robot.isFault():
            log.warn("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.clearFault()
            time.sleep(2)
            # Check again
            if robot.isFault():
                log.error("Fault cannot be cleared, exiting ...")
                return
            log.info("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()

        # Wait for the robot to become operational
        while not robot.isOperational():
            time.sleep(1)

        log.info("Robot is now operational")

        # Enable the robot, make sure the E-stop is released before enabling
        log.info("Enabling robot ...")
        robot.enable()
        while not robot.isOperational():
            time.sleep(1)
        log.info("Robot is now operational")

    except Exception as e:
        # Print exception error message
        log.error(str(e))


def move_home(robot):
    log.info("Moving to home pose")
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)
    robot.executePrimitive("Home()")
    # Wait for the primitive to finish
    while robot.isBusy():
        time.sleep(1)
    robot.executePrimitive("ZeroFTSensor()")


def get_robot_joints(robot) -> List[float]:
    """get robot current joints in rad

    Args:
            robot (_type_): _description_

    Returns:
            List[float]: joint positions
    """

    robot_states = flexivrdk.RobotStates()
    robot.getRobotStates(robot_states)
    return robot_states.q


if __name__ == "__main__":
    # robot_path = "/dev/ttyUSB0"
    # robot_overrides = "/home/rhos/Desktop/LeapTele/DexCopilot/isaacgymenvs/isaacgymenvs/tasks/lerobot/.cache/calibration/koch/left_leader.json"
    robot_path = "lerobot/configs/robot/koch.yaml"
    robot_overrides = ["~cameras", "~follower_arms"]
    detector = LeadArmReader(robot_path, robot_overrides)
    # ik
    # cam_agent = CameraD400(0)
    # cam_wrist = CameraD400(1)
    init_ee_state = None
    DOF = -1
    self_exam(log)
    move_home(robot)
    robot.setMode(mode.NRT_JOINT_POSITION)
    global gripper
    gripper = flexivrdk.Gripper(robot)
    try:
        gripper.move(0.09, 0.1, 20)
    except Exception as e:
        log.error(str(e))

    init_joints = get_robot_joints(robot)
    DOF = len(init_joints)
    target_vel = [0.0] * DOF
    target_acc = [0.0] * DOF
    MAX_VEL = [1.0] * DOF
    MAX_ACC = [0.5] * DOF

    dt = 1.0 / frequency
    init_joints = to_torch(init_joints)
    start_sample_time = time.time()
    while True:
        if robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")
        joints, gripper_angle = detector.get_follow_arm_joints()
        eef_position = detector.get_lead_arm_eef()
        lowcost_lower_bound = np.array([-0.09, 0.09, 0.02])
        lowcost_upper_bound = np.array([0.06, 0.24, 0.16])
        eef_pos = eef_position[0]
        outofbound = np.any(eef_pos > lowcost_upper_bound) or np.any(
            eef_pos < lowcost_lower_bound)
        # print("outofbound=", outofbound)
        hand_detected = (joints is not None) and (
            gripper_angle is not None) and not outofbound

        if robot.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")
        if time.time() - start_sample_time > dt:
            start_sample_time = time.time()
        if hand_detected:
            act_dof = torch.tensor(joints)
            finger_dis = gripper_angle

        # # print("finger_dis: ", finger_dis)
        # if finger_dis > 0.4:  # and env.cur_targets[:, DOF] < 0.03:
        #     sim_gripper_width = 0.10
        # elif finger_dis < 0.4:  # and env.cur_targets[:, DOF] > 0.08:
        #     sim_gripper_width = 0.03
        try:
            action = act_dof[:DOF].tolist()
            robot.sendJointPosition(
                action, target_vel, target_acc, MAX_VEL, MAX_ACC)
        except Exception as e:
            log.error(str(e))
