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
"""REAF Environment for the ideal robot setup."""

from collections.abc import Callable
import datetime

from dm_env import specs
import numpy as np
import ideal_robot
import ideal_robot_coordinator
import ideal_robot_device
import task_logic_elements
from reaf.common import environment_reset_from_callable
from reaf.common import maximum_steps_termination_checker
from reaf.common import partitioner_action_space_adapter
from reaf.common import time_trigger
from reaf.core import data_acquisition_and_control_layer
from reaf.core import default_discount_provider
from reaf.core import default_observation_space_adapter
from reaf.core import environment
from reaf.core import task_logic_layer


_EPISODE_PERIOD = datetime.timedelta(seconds=0.1)  # 100ms == 10Hz.


def create_environment(
    robot: ideal_robot.Robot,
    setpoint_provider: Callable[[], np.ndarray],
    reset_function: Callable[[], None],
    episode_length_in_seconds: float,
) -> environment.Environment:
  """Returns a new Environment for the specified robot."""

  # The DataAcquisitionAndControlLayer represents the "connection" to the
  # hardware. It collects measurements coming from the (hardware) sensors, sends
  # the correct control actions and synchronise the timing to the Environment
  # expected frequency.
  dacl = data_acquisition_and_control_layer.DataAcquisitionAndControlLayer(
      device_coordinator=ideal_robot_coordinator.IdealRobotCoordinator(robot),
      commands_trigger=None,
      measurements_trigger=time_trigger.TimeTrigger(period=_EPISODE_PERIOD),
  )

  # The TaskLogicLayer allows to customise the environment behaviour on a
  # per-task basis. Whilst the DataAcquisitionAndControlLayer is usually tied to
  # a specific hardware instantiation, the TaskLogicLayer can, and usually vary
  # depending on the actual task. It provides the possibility to add additional
  # "features" (i.e. produced values) that can be computed from the original
  # hardware measurements or from other sources and it provides the possibility
  # to modify the action provided by the agent to adapt it to the actual control
  # signal accepted by the hardware.
  # In addition to the above features it also allows to specify reward, discount
  # and episode termination criteria.

  # Adapt the agent action: the robot accepts currents but the environment
  # exposes torques.
  torques_to_current_processor = (
      task_logic_elements.ConstantFactorCommandsProcessor(
          consumed_spec=specs.BoundedArray(
              shape=(robot.dofs,),
              dtype=np.float32,
              minimum=-np.inf * np.ones(robot.dofs),
              maximum=np.inf * np.ones(robot.dofs),
          ),
          consumed_key="torque_reference",
          produced_key=ideal_robot_device.ROBOT_COMMAND_CURRENT_KEY,
          factor=ideal_robot.ROBOT_MOTOR_CONSTANT,
      )
  )

  # Specify a maximum duration for the episode.
  maximum_steps_termination = (
      maximum_steps_termination_checker.MaximumStepsTerminationChecker(
          max_steps=int(
              episode_length_in_seconds / _EPISODE_PERIOD.total_seconds()
          )
      )
  )

  ttl = task_logic_layer.TaskLogicLayer(
      commands_processors=(torques_to_current_processor,),
      features_producers=(
          task_logic_elements.ReferenceProducer(setpoint_provider),
      ),
      reward_provider=task_logic_elements.PositionAndVelocityErrorRewardProvider(
          position_reference_key=task_logic_elements.POSITION_REFERENCE_KEY,
          position_key=ideal_robot_device.ROBOT_MEASUREMENT_POSITION_KEY,
          velocity_key=ideal_robot_device.ROBOT_MEASUREMENT_VELOCITY_KEY,
      ),
      termination_checkers=(maximum_steps_termination,),
      discount_provider=default_discount_provider.DefaultDiscountProvider(),
      features_observers=(),
      loggers=(),
  )

  # Finally we put everything together into an Environment.

  # We first need to define how to map all the features produced by the
  # TaskLogicLayer (which, for example, can be used only for reward or
  # termination) into actual observations exposed by the Environment.
  #
  # In this example we return all the features, so we leave all parameters
  # except the first to None, but one can easily, e.g., remove the
  # "POSITION_REFERENCE_KEY" that was added by a feature producer as it is used
  # by the reward provided and we might not want to expose this to the
  # agent.
  observation_space_adapter = (
      default_observation_space_adapter.DefaultObservationSpaceAdapter(
          task_features_spec=ttl.features_spec(dacl.measurements_spec()),
          selected_features=None,
          renamed_features=None,
          observation_type_mapper=None,
      )
  )

  # We then need to define how the internal dict-based commands are mapped to
  # the environment actions.
  #
  # In this example (and this is a very common choice) the action is passed
  # directly from the agent and wrapped in a dictionary with the correct key.
  # Importantly, here is where we define the action_spec of the environment
  # which might be different from the one exposed by the
  # "DataAcquisitionAndControlLayer".

  # The action spec corresponds to the commands spec associated to the
  # ROBOT_COMMAND_CURRENT_KEY key.
  action_spec = dacl.commands_spec()[
      ideal_robot_device.ROBOT_COMMAND_CURRENT_KEY
  ]

  action_space_adapter = (
      partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
          partitions=(
              partitioner_action_space_adapter.PartitionInfo(
                  command_key="torque_reference"
              ),
          ),
          action_spec=action_spec,
      )
  )

  return environment.Environment(
      data_acquisition_and_control_layer=dacl,
      task_logic_layer=ttl,
      action_space_adapter=action_space_adapter,
      observation_space_adapter=observation_space_adapter,
      environment_reset=environment_reset_from_callable.EnvironmentResetFromCallable(
          lambda _: reset_function()
      ),
      end_of_episode_handler=None,
  )
