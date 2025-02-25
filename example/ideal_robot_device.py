# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""REAF device for the "IdealRobot"."""

from collections.abc import Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
import ideal_robot
from reaf.core import device
from typing_extensions import override


ROBOT_MEASUREMENT_POSITION_KEY = "joint_position"
ROBOT_MEASUREMENT_VELOCITY_KEY = "joint_velocity"
ROBOT_COMMAND_CURRENT_KEY = "current_reference"


class IdealRobotDevice(device.Device):
  """REAF device for the "IdealRobot"."""

  def __init__(self, robot: ideal_robot.Robot):
    self._robot = robot

  @override
  @property
  def name(self) -> str:
    return "IdealRobotDevice"

  @override
  def measurements_spec(self) -> dict[str, specs.Array]:
    return {
        ROBOT_MEASUREMENT_POSITION_KEY: specs.Array(
            shape=(self._robot.dofs,), dtype=np.float32
        ),
        ROBOT_MEASUREMENT_VELOCITY_KEY: specs.Array(
            shape=(self._robot.dofs,), dtype=np.float32
        ),
    }

  @override
  def commands_spec(self) -> dict[str, gdmr_types.AnyArraySpec]:
    return {
        ROBOT_COMMAND_CURRENT_KEY: specs.BoundedArray(
            shape=(self._robot.dofs,),
            dtype=np.float32,
            minimum=-np.inf * np.ones(self._robot.dofs),
            maximum=np.inf * np.ones(self._robot.dofs),
        )
    }

  @override
  def get_measurements(self) -> dict[str, np.ndarray]:
    position, velocity = self._robot.get_state()
    return {
        ROBOT_MEASUREMENT_POSITION_KEY: position.astype(np.float32),
        ROBOT_MEASUREMENT_VELOCITY_KEY: velocity.astype(np.float32),
    }

  @override
  def set_commands(self, commands: Mapping[str, np.ndarray]) -> None:
    currents = commands[ROBOT_COMMAND_CURRENT_KEY]
    self._robot.set_currents(currents)
