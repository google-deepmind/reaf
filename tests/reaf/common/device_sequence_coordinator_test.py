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
from unittest import mock

from reaf.common import device_sequence_coordinator
from reaf.core import device

from absl.testing import absltest


class DeviceSequenceCoordinatorTest(absltest.TestCase):

  def test_initialization_and_properties(self):
    mock_device1 = mock.create_autospec(device.Device, instance=True)
    mock_device2 = mock.create_autospec(device.Device, instance=True)
    mock_devices = [mock_device1, mock_device2]
    coordinator_name = "TestSequenceCoordinator"

    coordinator = device_sequence_coordinator.DeviceSequenceCoordinator(
        devices=mock_devices, name=coordinator_name
    )

    self.assertEqual(coordinator.name, coordinator_name)
    self.assertEqual(coordinator.get_devices(), mock_devices)


if __name__ == "__main__":
  absltest.main()
