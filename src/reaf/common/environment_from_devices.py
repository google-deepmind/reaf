# Copyright 2026 Google LLC
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
"""Creates an environment from a list of devices.

This module provides a function to create a REAF environment from a list of
devices. The environment will expose the devices as observations and commands,
and will provide a time measurement trigger with the specified control timestep.
Additionally, it will provide a maximum steps termination checker if
`episode_max_duration` is specified.
"""

from collections.abc import Callable, Sequence
import datetime

from reaf.common import device_sequence_coordinator
from reaf.common import environment_reset_from_callable
from reaf.common import maximum_steps_termination_checker
from reaf.common import time_trigger
from reaf.core import device
from reaf.core import device_layer as dl_module
from reaf.core import environment
from reaf.core import task_layer as tl_module


def environment_from_devices(
    devices: Sequence[device.Device],
    control_timestep: datetime.timedelta,
    episode_max_duration: datetime.timedelta | None = None,
    environment_reset: Callable[[], None] | None = None,
    coordinator_name: str = 'DeviceCoordinator',
) -> environment.Environment:
  """Creates an environment from a list of devices.

  This function creates a REAF environment with the following properties:
  - The DeviceLayer will expose the specified list of devices.
  - The DeviceLayer will provide a time measurement trigger with the specified
      control timestep.
  - The TaskLayer will not expose any additional command processor, feature
  producer
      or reward provider.
  - If `episode_max_duration` is specified, the environment will be configured
      to terminate an episode after the specified duration.
  - All the device measurements will be exposed as environment observations.
  - All the device commands will be exposed as environment actions.


  Args:
    devices: The list of devices to expose.
    control_timestep: The control timestep to use for the time measurement
      trigger.
    episode_max_duration: The maximum duration of an episode. If not specified,
      the episode will not terminate due to time limit.
    environment_reset: A callable that is called when the environment is reset.
    coordinator_name: The name of the device coordinator.

  Returns:
    The REAF environment.
  """
  device_layer = dl_module.DeviceLayer(
      device_coordinator=device_sequence_coordinator.DeviceSequenceCoordinator(
          devices=devices, name=coordinator_name
      ),
      commands_trigger=None,
      measurements_trigger=time_trigger.TimeTrigger(period=control_timestep),
  )

  termination_checkers = []
  if episode_max_duration is not None:
    termination_checkers.append(
        maximum_steps_termination_checker.MaximumStepsTerminationChecker(
            max_steps=episode_max_duration // control_timestep
        )
    )
  task_layer = tl_module.TaskLayer(
      commands_processors=(),
      features_producers=(),
      termination_checkers=termination_checkers,
      reward_provider=None,
      discount_provider=None,
  )

  # The environment reset takes a single argument (the options) but we don't
  # require it to the user to make this simpler.
  environment_reset_fn = (
      lambda _: environment_reset()
      if environment_reset is not None
      else lambda _: None
  )
  return environment.Environment(
      device_layer=device_layer,
      task_layer=task_layer,
      environment_reset=environment_reset_from_callable.EnvironmentResetFromCallable(
          environment_reset_fn  # pyrefly: ignore[bad-argument-type]
      ),
  )
