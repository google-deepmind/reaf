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
"""Codelab entry point.

This binary runs the example agent against the example environment in a runloop.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
from gdm_robotics.runtime import runloop as reaf_runloop
import numpy as np
import ideal_robot
import ideal_robot_device
import pd_agent
import robot_environment
import stdout_logger
import task_logic_elements

_NUM_EPISODES = flags.DEFINE_integer(
    "num_episodes", 5, "Number of episodes to run."
)
_EPISODE_LENGTH_S = flags.DEFINE_float(
    "episode_length", 10, "Episode length in seconds."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  num_dofs = 5
  # Create the robot here. In a real scenario this can represent a global
  # connection with the robot. Alternatevely the connection can be created
  # inside the REAF device.
  robot = ideal_robot.Robot(num_dofs=num_dofs)

  # In this example we create an agent exposing a PD control at the joint level
  # as control policy.
  agent = pd_agent.PdAgent(
      position_key=ideal_robot_device.ROBOT_MEASUREMENT_POSITION_KEY,
      velocity_key=ideal_robot_device.ROBOT_MEASUREMENT_VELOCITY_KEY,
      position_reference_key=task_logic_elements.POSITION_REFERENCE_KEY,
      p_diag_gains=np.array([1.0, 1.0, 0.7, 0.5, 0.3]),
      d_diag_gains=np.array([0.5, 0.5, 0.2, 0.1, 0.09]),
  )

  # We keep the setpoint constant throughout the episode, and we change it
  # during reset.
  current_setpoint = np.zeros(num_dofs)

  def _reset():
    logging.info("Resetting the episode")
    # Mark the setpoint as mutable.
    nonlocal current_setpoint

    # Randomise the initial position of the robot.
    position = np.random.uniform(low=-10, high=10, size=num_dofs)
    velocity = np.zeros(num_dofs)
    robot.reset_state(position, velocity)

    # Randomise the setpoint.
    current_setpoint = np.random.uniform(low=-8, high=8, size=num_dofs)

  # Expose a callable that returns the current setpoint. This will be provided
  # to the environment.
  def _get_current_setpoint() -> np.ndarray:
    return current_setpoint

  # Create the environment. We use an utility function to keep the main clearer.
  environment = robot_environment.create_environment(
      robot, _get_current_setpoint, _reset, _EPISODE_LENGTH_S.value
  )

  # Now that we have a policy and an environment we can create a runloop to
  # execute the system.
  runloop = reaf_runloop.Runloop(
      environment=environment,
      policy=agent,
      loggers=(stdout_logger.StdoutLogger(),),
  )
  # This will run for the specified number of episodes and then return.
  logging.info("Starting the runloop.")
  runloop.run(_NUM_EPISODES.value)
  logging.info("Runloop terminating. Releasing resources.")
  robot.shutdown()


if __name__ == "__main__":
  app.run(main)
