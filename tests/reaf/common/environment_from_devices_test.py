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
import datetime
from unittest import mock

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.common import environment_from_devices
from reaf.core import device
from reaf.core import numpy_mock_assertions

from absl.testing import absltest


class EnvironmentFromDevicesTest(absltest.TestCase):

  def test_devices_specs_are_the_environment_specs(self):
    # Mock devices with different specs
    mock_device1 = mock.create_autospec(device.Device, instance=True)
    mock_device1.name = "device1"
    mock_device1.measurements_spec.return_value = {
        "m1": specs.Array(shape=(1,), dtype=np.float64, name="m1"),
        "m2": specs.Array(shape=(2,), dtype=np.int32, name="m2"),
    }
    mock_device1.commands_spec.return_value = {
        "c1": gdmr_types.UnboundedArraySpec(
            shape=(1,), dtype=np.float32, name="c1"
        ),
    }

    mock_device2 = mock.create_autospec(device.Device, instance=True)
    mock_device2.name = "device2"
    mock_device2.measurements_spec.return_value = {
        "m3": specs.Array(shape=(3,), dtype=np.float32, name="m3"),
    }
    mock_device2.commands_spec.return_value = {
        "c2": gdmr_types.UnboundedArraySpec(
            shape=(2,), dtype=np.int32, name="c2"
        ),
        "c3": gdmr_types.UnboundedArraySpec(
            shape=(1,), dtype=np.float32, name="c3"
        ),
    }

    env = environment_from_devices.environment_from_devices(
        devices=[mock_device1, mock_device2],
        control_timestep=datetime.timedelta(seconds=1),
    )

    # Expected observation spec is the union of all measurement specs.
    expected_obs_spec = {
        "m1": specs.Array(shape=(1,), dtype=np.float64, name="m1"),
        "m2": specs.Array(shape=(2,), dtype=np.int32, name="m2"),
        "m3": specs.Array(shape=(3,), dtype=np.float32, name="m3"),
    }
    self.assertEqual(env.observation_spec(), expected_obs_spec)

    # Expected action spec is the union of all command specs.
    expected_action_spec = {
        "c1": gdmr_types.UnboundedArraySpec(
            shape=(1,), dtype=np.float32, name="c1"
        ),
        "c2": gdmr_types.UnboundedArraySpec(
            shape=(2,), dtype=np.int32, name="c2"
        ),
        "c3": gdmr_types.UnboundedArraySpec(
            shape=(1,), dtype=np.float32, name="c3"
        ),
    }
    self.assertEqual(env.action_spec(), expected_action_spec)

  def test_step_forwarding(self):
    mock_device1 = mock.create_autospec(device.Device, instance=True)
    mock_device1.name = "device1"
    mock_device1.measurements_spec.return_value = {
        "m1": specs.Array(shape=(1,), dtype=np.float32, name="m1"),
    }
    mock_device1.commands_spec.return_value = {
        "c1": gdmr_types.UnboundedArraySpec(
            shape=(1,), dtype=np.float32, name="c1"
        ),
    }
    mock_device1.get_measurements.return_value = {"m1": np.asarray([1.0])}
    mock_device1.set_commands.return_value = None

    mock_device2 = mock.create_autospec(device.Device, instance=True)
    mock_device2.name = "device2"
    mock_device2.measurements_spec.return_value = {
        "m2": specs.Array(shape=(2,), dtype=np.int32, name="m2"),
    }
    mock_device2.commands_spec.return_value = {
        "c2": gdmr_types.UnboundedArraySpec(
            shape=(2,), dtype=np.int32, name="c2"
        ),
    }
    mock_device2.get_measurements.return_value = {"m2": np.asarray([2, 3])}
    mock_device2.set_commands.return_value = None

    env = environment_from_devices.environment_from_devices(
        devices=[mock_device1, mock_device2],
        control_timestep=datetime.timedelta(seconds=1),
    )

    # Reset the environment to initialize
    timestep = env.reset()
    np.testing.assert_equal(
        timestep.observation,
        {
            "m1": np.asarray([1.0]),
            "m2": np.asarray([2, 3]),
        },
    )

    # Create a dummy action matching the env's action spec
    action = {
        "c1": np.asarray([10.0]).astype(np.float32),
        "c2": np.asarray([20, 30]).astype(np.int32),
    }

    timestep = env.step(action)

    # Assert apply_commands was called on each device with the relevant
    # sub-actions.
    numpy_mock_assertions.assert_called_once_with(
        mock_device1.set_commands, {"c1": np.asarray([10.0]).astype(np.float32)}
    )
    numpy_mock_assertions.assert_called_once_with(
        mock_device2.set_commands, {"c2": np.asarray([20, 30]).astype(np.int32)}
    )

    # Assert the observation contains the combined measurements
    np.testing.assert_equal(
        timestep.observation,
        {
            "m1": [1.0],
            "m2": [2, 3],
        },
    )

  def test_max_duration_termination(self):
    mock_device = mock.create_autospec(device.Device, instance=True)
    mock_device.name = "test_device"
    mock_device.measurements_spec.return_value = {}
    mock_device.commands_spec.return_value = {}
    mock_device.get_measurements.return_value = {}
    mock_device.set_commands.return_value = None

    control_timestep = datetime.timedelta(seconds=1)
    episode_max_duration = datetime.timedelta(seconds=3.5)
    # The MaximumStepsTerminationChecker uses
    # math.floor(episode_max_duration / control_timestep)
    # as the number of steps *after which* to terminate. So, max_steps = 3.
    # The episode will end after the 3rd call to step().

    env = environment_from_devices.environment_from_devices(
        devices=[mock_device],
        control_timestep=control_timestep,
        episode_max_duration=episode_max_duration,
    )

    # Reset the environment
    timestep = env.reset()
    self.assertTrue(timestep.first())

    # Step 1 to 2: Should be MID
    timestep = env.step({})
    self.assertTrue(timestep.mid())
    timestep = env.step({})
    self.assertTrue(timestep.mid())

    # Step 3: Should be LAST, as max_steps is 3.
    timestep = env.step({})
    self.assertTrue(timestep.last())
    # This is a truncation, which means that the discount should be 1.0.
    np.testing.assert_equal(timestep.discount, np.asarray(1.0))

    # Resetting should start a new episode.
    timestep = env.reset()
    self.assertTrue(timestep.first())

  def test_environment_reset_is_called(self):
    mock_device = mock.create_autospec(device.Device, instance=True)
    mock_device.name = "test_device"
    mock_device.measurements_spec.return_value = {}
    mock_device.commands_spec.return_value = {}
    mock_device.get_measurements.return_value = {}
    mock_device.set_commands.return_value = None

    mock_reset_fn = mock.Mock()

    env = environment_from_devices.environment_from_devices(
        devices=[mock_device],
        control_timestep=datetime.timedelta(seconds=1),
        environment_reset=mock_reset_fn,
    )

    # Reset the environment
    env.reset()

    # Assert that the mock reset function was called.
    mock_reset_fn.assert_called_once()

    # Reset again
    env.reset()
    # Assert it was called a second time.
    self.assertEqual(mock_reset_fn.call_count, 2)


if __name__ == "__main__":
  absltest.main()
