#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import dataclass, field, replace

import torch
import numpy as np
from typing import List
import flexivrdk

from lerobot.common.robot_devices.cameras.utils import Camera
from lerobot.common.robot_devices.motors.utils import MotorsBus
from lerobot.common.robot_devices.MiniTeleop_utils import LeadArmReader


@dataclass
class FlexivRobotConfig:
    robot_type: str | None = "flexiv"
    leader_arm: dict[str, str] = field(default_factory=lambda: {})
    ips: dict[str, str] = field(default_factory=lambda: {})
    cameras: dict[str, Camera] = field(default_factory=lambda: {})
    # TODO(aliberts): add feature with max_relative target
    # TODO(aliberts): add comment on max_relative target
    max_relative_target: list[float] | float | None = None
    gripper_open_degree: float | None = None


class FlexivRobot():
    """Wrapper of stretch_body.robot.Robot"""

    def __init__(self, config: FlexivRobotConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = FlexivRobotConfig()
        # Overwrite config arguments using kwargs
        self.config = replace(config, **kwargs)

        self.robot_type = self.config.robot_type
        self.leader_arm = self.config.leader_arm
        self.ips = self.config.ips
        self.cameras = self.config.cameras
        self.is_connected = False
        self.teleop = None
        self.logs = {}
        self.log = flexivrdk.Log()

        self.state_keys = None
        self.action_keys = None
        self.DOF = None
        self.target_vel = None
        self.target_acc = None
        self.MAX_VEL = None
        self.MAX_ACC = None
        self.flexiv = None
        self.gripper = None

    def connect(self) -> None:
        # connect flexiv
        try:
            self.flexiv = flexivrdk.Robot(
                self.ips["robot_ip"], self.ips["local_ip"])
            self.gripper = flexivrdk.Gripper(self.flexiv)
            self.is_connected = self.flexiv.isFault() == False

            if not self.is_connected:
                self.log.warn(
                    "Fault occurred on robot server, trying to clear ...")
                # Try to clear the fault
                self.flexiv.clearFault()
                time.sleep(2)
                # Check again
                if self.flexiv.isFault():
                    self.log.error("Fault cannot be cleared, exiting ...")
                    raise ConnectionError()
                self.log.info("Fault on robot server is cleared")

            # Enable the robot, make sure the E-stop is released before enabling
            self.log.info("Enabling robot ...")
            self.flexi.enable()

            # Wait for the robot to become operational
            while not self.flexi.isOperational():
                time.sleep(1)

            self.log.info("Robot is now operational")

            # Enable the robot, make sure the E-stop is released before enabling
            self.log.info("Enabling robot ...")
            self.flexi.enable()
            while not self.flexi.isOperational():
                time.sleep(1)
            self.log.info("Robot is now operational")

            init_joints = self.get_state(self.flexiv)
            DOF = len(init_joints)
            self.target_vel = [0.0] * DOF
            self.target_acc = [0.0] * DOF
            self.MAX_VEL = [1.0] * DOF
            self.MAX_ACC = [0.5] * DOF
            self.DOF = DOF

        except Exception as e:
            # Print exception error message
            self.log.error(str(e))

        # connect leader arm
        robot_overrides = ["~cameras", "~follower_arms"]
        self.teleop = LeadArmReader(
            self.leader_arm["robot_path"], robot_overrides, visualize=self.leader_arm["visualize"])

        # connect cameras
        for name in self.cameras:
            self.cameras[name].connect()
            self.is_connected = self.is_connected and self.cameras[name].is_connected

        if not self.is_connected:
            print(
                "Could not connect to the cameras, check that all cameras are plugged-in.")
            raise ConnectionError()

        self.home()

    def home(self) -> None:
        mode = flexivrdk.Mode
        self.log.info("Moving to home pose")
        self.flexiv.setMode(mode.NRT_PRIMITIVE_EXECUTION)
        self.flexiv.executePrimitive("Home()")
        # Wait for the primitive to finish
        while self.flexiv.isBusy():
            time.sleep(1)
        self.flexiv.executePrimitive("ZeroFTSensor()")

    def set_robot_home_position(self) -> None:
        self.home()

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        if self.teleop is None:
            print("teleop is None")
            raise ConnectionError()

        before_read_t = time.perf_counter()
        state = self.get_state()
        # action = self.teleop.gamepad_controller.get_state()
        action, gripper_angle = self.teleop.get_follow_arm_joints()
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        before_write_t = time.perf_counter()

        eef_position = self.teleop.get_lead_arm_eef()
        lowcost_lower_bound = np.array([-0.09, 0.09, 0.02])
        lowcost_upper_bound = np.array([0.06, 0.24, 0.16])
        eef_pos = eef_position[0]
        outofbound = np.any(eef_pos > lowcost_upper_bound) or np.any(
            eef_pos < lowcost_lower_bound)
        # print("outofbound=", outofbound)
        action_valid = (action is not None) and (
            gripper_angle is not None) and not outofbound

        if self.flexiv.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if action_valid:
            act_dof = torch.tensor(action)
            finger_dis = gripper_angle
            try:
                action = act_dof[:self.DOF].tolist()
                self.flexiv.sendJointPosition(
                    action, self.target_vel, self.target_acc, self.MAX_VEL, self.MAX_ACC)
            except Exception as e:
                self.log.error(str(e))

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # if self.state_keys is None:
        #     self.state_keys = list(state)

        if not record_data:
            return

        state = torch.as_tensor(list(state))
        action = torch.as_tensor(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter(
            ) - before_camread_t

        # Populate output dictionnaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def to_torch(x, dtype=torch.float, device="cuda:0", requires_grad=False):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

    def get_state(self) -> List[float]:
        robot_states = flexivrdk.RobotStates()
        self.flexiv.getRobotStates(robot_states)
        return robot_states.q

    def capture_observation(self) -> dict:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        # before_read_t = time.perf_counter()
        state = self.get_state()
        # self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        # if self.state_keys is None:
        #     self.state_keys = list(state)

        state = torch.as_tensor(list(state))

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter(
            ) - before_camread_t

        # Populate output dictionnaries
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        # TODO(aliberts): return ndarrays instead of torch.Tensors
        if not self.is_connected:
            raise ConnectionError()

        # if self.teleop is None:
        #     self.teleop = GamePadTeleop(robot_instance=False)
        #     self.teleop.startup(robot=self)

        # if self.action_keys is None:
        #     dummy_action = self.teleop.gamepad_controller.get_state()
        #     self.action_keys = list(dummy_action.keys())

        action = action[:self.DOF]
        gripper_angle = action[-1]

        before_write_t = time.perf_counter()

        eef_position = self.teleop.get_lead_arm_eef()
        lowcost_lower_bound = np.array([-0.09, 0.09, 0.02])
        lowcost_upper_bound = np.array([0.06, 0.24, 0.16])
        eef_pos = eef_position[0]
        outofbound = np.any(eef_pos > lowcost_upper_bound) or np.any(
            eef_pos < lowcost_lower_bound)
        # print("outofbound=", outofbound)
        action_valid = (action is not None) and (
            gripper_angle is not None) and not outofbound

        if self.flexiv.isFault():
            raise Exception("Fault occurred on robot server, exiting ...")

        if action_valid:
            act_dof = torch.tensor(action)
            finger_dis = gripper_angle
            try:
                action = act_dof[:self.DOF].tolist()
                self.flexiv.sendJointPosition(
                    action, self.target_vel, self.target_acc, self.MAX_VEL, self.MAX_ACC)
            except Exception as e:
                self.log.error(str(e))

        self.logs["write_pos_dt_s"] = time.perf_counter() - before_write_t

        # TODO(aliberts): return action_sent when motion is limited
        return action

    def print_logs(self) -> None:
        pass
        # TODO(aliberts): move robot-specific logs logic here

    def teleop_safety_stop(self) -> None:
        if self.teleop is not None:
            self.teleop._safety_stop(robot=self)

    def disconnect(self) -> None:
        self.stop()
        if self.teleop is not None:
            self.teleop.gamepad_controller.stop()
            self.teleop.stop()

        if len(self.cameras) > 0:
            for cam in self.cameras.values():
                cam.disconnect()

        self.is_connected = False

    def __del__(self):
        self.disconnect()
