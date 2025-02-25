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
"""Coordinator for the IdealRobot setup."""

from collections.abc import Iterable
import ideal_robot
import ideal_robot_device
from reaf.core import device
from reaf.core import device_coordinator
from typing_extensions import override


class IdealRobotCoordinator(device_coordinator.DeviceCoordinator):
  """Device coordinator for the "IdealRobot" setup."""

  def __init__(self, robot: ideal_robot.Robot):
    self._device = ideal_robot_device.IdealRobotDevice(robot=robot)

  @override
  @property
  def name(self) -> str:
    return "environment_coordinator"

  @override
  def get_devices(self) -> Iterable[device.Device]:
    return (self._device,)
