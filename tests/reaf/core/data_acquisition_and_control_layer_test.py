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

from dm_env import specs
import numpy as np
from reaf.core import data_acquisition_and_control_layer
from reaf.core import device as reaf_device
from reaf.core import device_coordinator as reaf_coordinator
from reaf.core import trigger

from absl.testing import absltest


class DataAcquisitionControlLayerTest(absltest.TestCase):

  def test_commands_spec_are_correct(self):
    device1 = mock.create_autospec(reaf_device.Device, instance=True)
    device2 = mock.create_autospec(reaf_device.Device, instance=True)

    device1_spec = {
        'c1/command1': specs.Array(shape=(1,), dtype=np.float32),
        'c1/command2': specs.BoundedArray(
            shape=(2,), dtype=np.int32, minimum=-10, maximum=10
        ),
    }
    device2_spec = {
        'c2/command1': specs.StringArray(shape=()),
        'c2/command2': specs.DiscreteArray(num_values=4),
    }

    device1.commands_spec.return_value = device1_spec
    device2.commands_spec.return_value = device2_spec

    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )
    coordinator.get_devices.return_value = (device1, device2)

    dacl = data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
        device_coordinator=coordinator,
        commands_trigger=None,
        measurements_trigger=None,
    )

    self.assertEqual(device1_spec | device2_spec, dacl.commands_spec())

  def test_measurements_spec_are_correct(self):
    device1 = mock.create_autospec(reaf_device.Device, instance=True)
    device2 = mock.create_autospec(reaf_device.Device, instance=True)

    device1_spec = {
        'c1/measurements1': specs.Array(shape=(1,), dtype=np.float32),
        'c1/measurements2': specs.BoundedArray(
            shape=(2,), dtype=np.int32, minimum=-10, maximum=10
        ),
    }
    device2_spec = {
        'c2/measurements1': specs.StringArray(shape=()),
        'c2/measurements2': specs.DiscreteArray(num_values=4),
    }

    device1.measurements_spec.return_value = device1_spec
    device2.measurements_spec.return_value = device2_spec

    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )
    coordinator.get_devices.return_value = (device1, device2)

    dacl = data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
        device_coordinator=coordinator,
        commands_trigger=None,
        measurements_trigger=None,
    )

    self.assertEqual(device1_spec | device2_spec, dacl.measurements_spec())

  def test_overlapping_commands_spec_keys_raise_error(self):
    device1 = mock.create_autospec(reaf_device.Device, instance=True)
    device2 = mock.create_autospec(reaf_device.Device, instance=True)

    # A collision on the key is enough to raise an error.
    device1_spec = {
        'c1/command1': specs.Array(shape=(1,), dtype=np.float32),
        'c1/command2': specs.BoundedArray(
            shape=(2,), dtype=np.int32, minimum=-10, maximum=10
        ),
    }
    device2_spec = {
        'c2/command1': specs.StringArray(shape=()),
        'c1/command2': specs.DiscreteArray(num_values=4),
    }

    device1.commands_spec.return_value = device1_spec
    device2.commands_spec.return_value = device2_spec

    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )
    coordinator.get_devices.return_value = (device1, device2)

    with self.assertRaises(RuntimeError):
      data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
          device_coordinator=coordinator,
          commands_trigger=None,
          measurements_trigger=None,
      )

  def test_overlapping_measurements_spec_keys_raise_error(self):
    device1 = mock.create_autospec(reaf_device.Device, instance=True)
    device2 = mock.create_autospec(reaf_device.Device, instance=True)

    # A collision on the key is enough to raise an error.
    device1_spec = {
        'c1/measurements1': specs.Array(shape=(1,), dtype=np.float32),
        'c1/measurements2': specs.BoundedArray(
            shape=(2,), dtype=np.int32, minimum=-10, maximum=10
        ),
    }
    device2_spec = {
        'c2/measurements1': specs.StringArray(shape=()),
        'c1/measurements2': specs.DiscreteArray(num_values=4),
    }

    device1.measurements_spec.return_value = device1_spec
    device2.measurements_spec.return_value = device2_spec

    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )
    coordinator.get_devices.return_value = (device1, device2)

    with self.assertRaises(RuntimeError):
      data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
          device_coordinator=coordinator,
          commands_trigger=None,
          measurements_trigger=None,
      )

  def test_episode_calls_devices(self):
    device1 = mock.create_autospec(reaf_device.Device, instance=True)
    device2 = mock.create_autospec(reaf_device.Device, instance=True)

    # We need to specify the commands_spec as the map is built during the DACL
    # initialisation.

    device1_spec = {
        'c1/command1': specs.Array(shape=(1,), dtype=np.float32),
    }
    device2_spec = {
        'c2/command1': specs.Array(shape=(2,), dtype=np.int32),
    }

    device1.commands_spec.return_value = device1_spec
    device2.commands_spec.return_value = device2_spec

    # Arbitrary values as measurements.

    c1_initial_value = np.asarray(1.1)
    c1_value = c1_initial_value

    def _c1_measurements():
      nonlocal c1_value
      c1_value = np.square(c1_value)
      return {'c1/measurements1': np.asarray(c1_value)}

    c2_value_initial_value = np.array([-1, 3])
    c2_value = c2_value_initial_value

    def _c2_measurements():
      nonlocal c2_value
      increment = np.asarray([1, -1])
      c2_value = c2_value + increment
      return {'c2/measurements1': c2_value}

    device1.get_measurements.side_effect = _c1_measurements
    device2.get_measurements.side_effect = _c2_measurements

    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )
    coordinator.get_devices.return_value = (device1, device2)

    dacl = data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
        device_coordinator=coordinator,
        commands_trigger=None,
        measurements_trigger=None,
    )

    all_measurements = []
    # Arbitrary values.
    commands = [
        {'c1/command1': np.array([4.32]), 'c2/command1': np.array([1, 2])},
        {'c1/command1': np.array([-0.32]), 'c2/command1': np.array([4, 6])},
        {'c1/command1': np.array([1.32]), 'c2/command1': np.array([-3, 1])},
        {'c1/command1': np.array([3.32]), 'c2/command1': np.array([-16, 4])},
    ]

    all_measurements.append(dacl.begin_stepping())
    for command in commands:
      all_measurements.append(dacl.step(command))
    dacl.end_stepping()

    coordinator.on_begin_stepping.assert_called_once()
    coordinator.on_end_stepping.assert_called_once()

    # Abuse of assertLen, i.e. we are not really asserting a container len.
    self.assertLen(commands, coordinator.before_set_commands.call_count)
    self.assertLen(commands, coordinator.after_set_commands.call_count)
    self.assertEqual(
        len(commands) + 1, coordinator.before_get_measurements.call_count
    )

    # We can't use assert_has_calls with numpy arrays as mock uses the
    # == operator which returns an array for numpy arrays.

    # Check the length first.
    self.assertLen(commands, device1.set_commands.call_count)
    self.assertLen(commands, device2.set_commands.call_count)

    for index, call in enumerate(device1.set_commands.mock_calls):
      sub_command = {'c1/command1': commands[index]['c1/command1']}
      # call is a tuple. In this case, second element is the positional
      # argument, returned as a tuple. We are interested in the only argument.
      call_argument = call[1][0]
      np.testing.assert_equal(sub_command, call_argument)

    for index, call in enumerate(device2.set_commands.mock_calls):
      sub_command = {'c2/command1': commands[index]['c2/command1']}
      # call is a tuple. In this case, second element is the positional
      # argument, returned as a tuple. We are interested in the only argument.
      call_argument = call[1][0]
      np.testing.assert_equal(sub_command, call_argument)

    # Check that measurements are correct.
    self.assertEqual(len(commands) + 1, device1.get_measurements.call_count)
    self.assertEqual(len(commands) + 1, device2.get_measurements.call_count)

    # Reset the value used by the functions.
    c1_value = c1_initial_value
    c2_value = c2_value_initial_value

    expected_measurements = []
    for _ in range(len(commands) + 1):
      measurements = {}
      measurements.update(_c1_measurements())
      measurements.update(_c2_measurements())
      expected_measurements.append(measurements)

    np.testing.assert_equal(expected_measurements, all_measurements)

  def test_triggers_are_called(self):
    device1 = mock.create_autospec(reaf_device.Device, instance=True)
    device2 = mock.create_autospec(reaf_device.Device, instance=True)

    command_trigger = mock.create_autospec(trigger.Trigger, instance=True)
    measurements_trigger = mock.create_autospec(trigger.Trigger, instance=True)

    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )
    coordinator.get_devices.return_value = (device1, device2)

    dacl = data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
        device_coordinator=coordinator,
        commands_trigger=command_trigger,
        measurements_trigger=measurements_trigger,
    )

    dacl.begin_stepping()
    measurements_trigger.wait_for_event.assert_called_once()
    command_trigger.wait_for_event.assert_not_called()

    measurements_trigger.wait_for_event.reset_mock()
    command_trigger.wait_for_event.reset_mock()

    # Arbitrary number of calls.
    for _ in range(10):
      dacl.step({})

    self.assertEqual(10, measurements_trigger.wait_for_event.call_count)
    self.assertEqual(10, command_trigger.wait_for_event.call_count)

  def test_returns_coordinator(self):
    # Create a coordinator.
    coordinator = mock.create_autospec(
        reaf_coordinator.DeviceCoordinator, instance=True
    )

    # Create dacl and test that it returns the coordinator.
    dacl = data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
        device_coordinator=coordinator,
        commands_trigger=None,
        measurements_trigger=None,
    )
    self.assertEqual(coordinator, dacl.device_coordinator)


if __name__ == '__main__':
  absltest.main()
