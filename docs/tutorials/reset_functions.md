# Tutorial: Mastering Complex Resets

In our previous tutorials, we focused on the `step()` function, which defines
the environment's behavior during an episode. In this tutorial, we will dive
deep into the other critical part of an environment's lifecycle: the
**`reset()`** function.

We will start with a general overview of resets in Reinforcement Learning and
then explore the powerful and flexible tools REAF provides for handling even the
most complex reset procedures in robotics.

--------------------------------------------------------------------------------

## The Role of `reset` in Reinforcement Learning

In a standard Reinforcement Learning (RL) loop, the `reset()` function marks the
boundary between episodes. Think of it like hitting the "restart" button on a
level in a video game. When an episode ends—either because the agent succeeded,
failed, or a time limit was reached—the environment needs to be returned to a
valid starting state for the next attempt.

In a typical RL framework like `dm_env`, the `reset()` method has two
fundamental jobs:

1.  **Reinitialize the State**: It brings the world back to a valid starting
    position. This could be the same position every time or a randomized one to
    help the agent generalize.
2.  **Return the First Observation**: It provides the agent with the very first
    observation of the new episode, allowing it to choose its first action.

While this concept is simple, a reset in a real robotics environment can be a
very complex operation, involving multiple coordinated steps. This is where
REAF's structured approach comes in.

--------------------------------------------------------------------------------

## How REAF Handles Resets

REAF acknowledges that a robotic reset is more than just resetting a few
variables. You might need to physically move a robot arm, call an external
simulation service, or randomize objects in a bin. To manage this complexity,
REAF provides a powerful two-level system for handling resets.

### 1. Automatic Resets for Task Layer Components

As we saw in the previous tutorial with our `RelativePositionFeaturesProducer`,
individual components within the **Task Layer** can be stateful.
That `FeaturesProducer` needed to store the robot's initial position at the
start of each episode.

REAF handles this automatically. When the environment's main `reset()` is
called, REAF iterates through **all components in the Task Layer**
(`FeaturesProducer`s, `CommandsProcessor`s, etc.) and calls their individual
`reset()` methods. This allows each component to cleanly manage its own internal
state at the beginning of an episode without any extra effort from you.

### 2. Full Control with `EnvironmentReset`

For high-level, coordinated reset logic that involves the entire environment,
REAF provides the **`EnvironmentReset`** class.

This is a special object that you can create and pass to the environment to
define a custom, multi-step reset procedure. The implementation is completely
free and up to you, giving you total control. You would use a custom
`EnvironmentReset` object to handle complex tasks such as:

*   Physically moving a robot arm to a "home" position before starting a task.
*   Randomizing the locations of objects in a scene for curriculum learning.
*   Calling an external API to reset a simulation or physics engine.
*   Loading a new scene or configuration from a file.

By separating the automatic, low-level state management of individual components
from the high-level, user-defined reset orchestration, REAF provides a system
that is both easy to use for simple cases and powerful enough for complex
robotics research.

## Example: A Randomized Reset Handler

Now, let's create a practical example of a custom `EnvironmentReset`. Our goal
is to create a reset handler that repositions the robot to a new, slightly
randomized position at the start of each episode. This is a common technique
used in RL to train more robust policies that can generalize beyond a single
starting condition.

Our reset handler will: 1. Take a "home" position and a randomization amount as
input. 2. At each reset, generate a new random position centered around the home
position. 3. Call our `ideal_robot.Robot`'s `reset_state()` method with the new
position and zero velocity.

--------------------------------------------------------------------------------

### Step 1: Implementing the `EnvironmentReset` Class

We'll start by creating a new class that inherits from
`environment_reset.EnvironmentReset`. The core logic goes into the `do_reset`
method, which is called automatically by the environment's public `reset()`
function.

```python
import logging
import numpy as np
from typing_extensions import override

import .ideal_robot
from reaf.core import environment
from gdm_robotics.interfaces import environment as gdmr_env

class RandomizedRobotReset(environment.EnvironmentReset):
  """A custom reset handler that resets the robot to a random position."""

  def __init__(
      self,
      robot: ideal_robot.Robot,
      home_position: np.ndarray,
      randomization_scale: float,
  ):
    """
    Args:
      robot: An instance of the robot API to call its reset method.
      home_position: The center of the randomization distribution.
      randomization_scale: The maximum displacement from the home position.
    """
    self._robot = robot
    self._home_position = home_position
    self._randomization_scale = randomization_scale

  @override
  def do_reset(
      self, options: gdmr_env.ResetOptions
  ) -> None:
    """Implements the custom reset logic."""
    # 1. Generate random noise, scaled by our randomization factor.
    noise = (
        np.random.uniform(-1.0, 1.0, size=self._robot.dofs)
        * self._randomization_scale
    )

    # 2. Calculate the new target position.
    target_position = self._home_position + noise

    # 3. Create a zero-velocity vector.
    target_velocity = np.zeros(self._robot.dofs)

    logging.info("Resetting robot to randomized position: %s", target_position)

    # 4. Call the underlying robot's API to physically reset it.
    self._robot.reset_state(
        new_position=target_position, new_velocity=target_velocity
    )
```

### Step 2: Integrating with the Environment

Now that we have our custom reset class, the final step is to tell our REAF
environment to use it. We do this by passing an instance of
`RandomizedRobotReset` to the `environment.Environment` constructor via the
`environment_reset` argument.

Here is the updated `build_custom_environment` function from the previous
tutorial, with the new lines highlighted:

```py
def build_custom_environment(
    robot: ideal_robot.Robot,
    left_camera: ideal_camera.RgbCamera,
    right_camera: ideal_camera.RgbCamera,
    max_duration: datetime.timedelta,
    control_timestep: datetime.timedelta,
) -> environment.Environment:
  """Builds our custom environment, now with a randomized reset."""

  # ... (Steps 1 and 2 for building the Device Layer and Task Layer are the same as before)
  # ...

  # 3. Build the final Environment, now including our custom reset handler.

  # Instantiate the reset handler.
  home_position = np.zeros(robot.dofs)
  reset_handler = RandomizedRobotReset(
      robot=robot, home_position=home_position, randomization_scale=0.1
  )
  # Pass the reset handler to the environment's constructor.
  return environment.Environment(
      device_layer=device_layer,
      task_layer=task_layer,
      environment_reset=reset_handler,   # <--- THIS IS THE NEW PART
  )
```

And that's it! When you call `env.reset()`, our `RandomizedRobotReset` logic
will be executed, providing a different starting condition for every episode.
This is the fundamental pattern for managing complex, stateful resets in REAF.
