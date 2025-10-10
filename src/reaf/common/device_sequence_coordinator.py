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
"""A device coordinator that manages a sequence of devices."""

from collections.abc import Iterable, Sequence

from reaf.core import device
from reaf.core import device_coordinator


class DeviceSequenceCoordinator(device_coordinator.DeviceCoordinator):
  """A device coordinator that manages a sequence of devices."""

  def __init__(self, devices: Sequence[device.Device], name: str):
    self._devices = devices
    self._name = name

  @property
  def name(self) -> str:
    """Returns the name of the coordinator."""
    return self._name

  def get_devices(self) -> Iterable[device.Device]:
    """Returns the devices composing the embodiment."""
    return self._devices
