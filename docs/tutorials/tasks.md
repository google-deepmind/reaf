# Tutorial: Customizing with the Task Layer

In the last tutorial, we built `Device`s that directly exposed a robot's
hardware interface. Now, we'll learn how to build on top of that using the
**Task Layer**.

The Task Layer is where you define the "rules of the game" for your specific task. It
sits between the agent and the hardware, transforming data in both directions to
bridge the gap between high-level goals and low-level control. It's designed to
be highly modular, allowing you to easily swap out components and experiment
with different task variations.

In this tutorial, we will focus on the two most common Task Layer components:

*   **`CommandsProcessor`**: Intercepts and modifies the agent's `action`
    *before* it's sent to the hardware.
*   **`FeaturesProducer`**: Takes the raw `measurements` from the hardware and
    creates new, more meaningful `features` for the agent.

--------------------------------------------------------------------------------

## Our Goal for This Tutorial 🎯

We will customize our simple environment from the last tutorial to meet two new
requirements:

1.  **Our agent will now output torque commands**, but our simulated hardware
    only accepts `current` commands. We will build a **`CommandsProcessor`** to
    handle this conversion.
2.  **We want the agent to see joint positions relative to a "zero" position**
    that is recorded at the start of each episode. We will build a
    **`FeaturesProducer`** to compute this on the fly.

--------------------------------------------------------------------------------

## ⚠️ A Note on Performance and Control Frequency

It's critical to remember that all processing inside the Task Layer happens
**synchronously** within the environment's `step()` method. This means your Task Layer
logic is executed at the same frequency as the agent's policy (the "control
frequency").

Any processing that needs to run faster than the agent's control loop - such as
a low-level torque controller or a high-frequency safety monitor - **cannot be
implemented in a simple Python processor**.

Instead, these high-frequency operations should be offloaded to a separate
context, such as: * A background thread (ideally implemented in a
high-performance language like C++). * The robot's underlying middleware or
hardware controller.

The Task Layer is best reserved for operations that are synchronous with the agent's
decision-making, like converting action spaces or computing task-level features
and rewards.

--------------------------------------------------------------------------------

## Step 1: Converting Actions with a `CommandsProcessor`

Our first goal is to bridge the gap between our agent's **torque** actions and
our hardware's **current** commands. This is a perfect job for a
**`CommandsProcessor`**.

A `CommandsProcessor` is a Task Layer component that intercepts and transforms the
dictionary of commands as it flows from the agent to the hardware. Its core job
is to **consume** one or more commands from the dictionary and **produce** one
or more new ones in their place.

For our use case, the transformation will look like this: * **Input
(Consumed):** A dictionary containing `{'agent_torque': ...}` * **Output
(Produced):** A new dictionary containing `{'current_reference': ...}`

Since torque is proportional to current for a DC motor (${\tau = K \cdot i}$),
we can implement this with a simple processor that multiplies the incoming
command by a constant factor.

### The `ConstantFactorCommandsProcessor`

Let's build it. The class needs to do three things: 1. Implement the core logic
in `process_commands`. 2. Declare which command it consumes via
`consumed_commands_spec`. 3. Declare which command it produces via
`produced_commands_keys`.

Here is the complete implementation:

```python
from collections.abc import Mapping
from typing_extensions import override
import numpy as np

from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import commands_processor

class ConstantFactorCommandsProcessor(commands_processor.CommandsProcessor):
  """Multiplies a consumed command by a factor to produce a new one."""

  def __init__(self,
               produced_key: str,
               consumed_key: str,
               consumed_spec: gdmr_types.AnyArraySpec,
               constant_factor: float):
    """
    Args:
      produced_key: The key of the new command to add to the dictionary.
      consumed_key: The key of the command to remove from the dictionary.
      consumed_spec: The spec of the consumed command.
      constant_factor: The multiplicative factor to apply.
    """
    self._produced_key = produced_key
    self._consumed_key = consumed_key
    self._consumed_spec = consumed_spec
    self._constant_factor = constant_factor

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    """The core logic: multiply the consumed command by the factor."""
    original_command = consumed_commands[self._consumed_key]
    final_command = original_command * self._constant_factor
    return {self._produced_key: final_command}

  @override
  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    """Declares the spec of the command this processor will consume."""
    # This tells REAF to remove the command matching this spec from the main
    # command dictionary and pass it to `process_commands`.
    return {self._consumed_key: self._consumed_spec}

  @override
  def produced_commands_keys(self) -> set[str]:
    """Declares the keys of the commands this processor produces."""
    # This tells REAF to take the dictionary returned by `process_commands`
    # and merge it back into the main command dictionary.
    return {self._produced_key}
```

**Design Note**: You may notice an asymmetry: we provide the full `spec` for
consumed commands but only the `key` for produced commands. This is intentional.
The spec of a produced command is defined by the downstream component that will
eventually consume it (in our case, the Device Layer). This design avoids redundant spec
definitions.

## Step 2: Creating Relative Features with a `FeaturesProducer`

Our second goal is to provide the agent with the robot's joint positions
relative to its starting configuration in the episode. The raw measurement from
our `Device` is always the *absolute* position. We need a way to store the
initial position at the start of an episode and then, at every step, calculate
the difference.

This is the perfect job for a **`FeaturesProducer`**.

A `FeaturesProducer` is a Task Layer component that takes the existing dictionary of
measurements and features and **produces** a new, enriched dictionary containing
additional features. Unlike `CommandsProcessor`s, which are mutable,
`FeaturesProducer`s are **immutable**—they can only *add* new features, not
modify or remove existing ones.

### The `RelativePositionFeaturesProducer`

To accomplish our goal, our `FeaturesProducer` needs to be **stateful**. It
will: 1. Implement the `reset()` method to capture and store the initial joint
positions at the beginning of an episode. 2. Implement the `produce_features()`
method to calculate the delta between the current position and the stored
initial position at every step. 3. Declare its dependencies and outputs using
`required_features_keys()` and `produced_features_spec()`.

Here is the complete implementation:

```python
from collections.abc import Mapping
from typing_extensions import override
import numpy as np

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import features_producer

# Define constants for our feature keys. In this example we are hardcoding them
# as constants but otherwise this can be provided as arguments to the producer.
ABSOLUTE_POSITION_KEY = "joint_position"  # This is the input from the Device Layer.
RELATIVE_POSITION_KEY = "relative_joint_position"  # This is our new output.

class RelativePositionFeaturesProducer(features_producer.FeaturesProducer):
  """A stateful producer that calculates position relative to an initial pose."""

  def __init__(self, position_spec: gdmr_types.AnyArraySpec):
    """
    Args:
      position_spec: The spec for the produced relative joint position feature.
    """
    self._position_spec = position_spec
    # This will store the robot's pose at the start of the episode.
    self._initial_position: np.ndarray | None = None

  @override
  def reset(self) -> None:
    """Called at the start of an episode to store the initial position."""
    # Set to None the initial position. This will trigger saving it at the first
    # call to `produce_features`.
    self._initial_position = None

  @override
  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    """Called at every step to calculate and produce the new feature."""
    current_position = required_features[ABSOLUTE_POSITION_KEY]

    if self._initial_position is None:
      self._initial_position = np.array(current_position)

    relative_position = current_position - self._initial_position

    # Return a dictionary containing only the new feature(s).
    return {RELATIVE_POSITION_KEY: relative_position}

  @override
  def required_features_keys(self) -> set[str]:
    """Declares that this producer needs the absolute position to work."""
    return {ABSOLUTE_POSITION_KEY}

  @override
  def produced_features_spec(
      self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    """Declares the spec of the new relative position feature."""
    return {RELATIVE_POSITION_KEY: self._position_spec}
```

By implementing both reset() and produce_features(), we have successfully
created a stateful Task Layer component that enriches the data stream with a new,
task-relevant feature.

## Step 3: Assembling the Custom Environment

We have our `Device`, our `CommandsProcessor`, and our `FeaturesProducer`. Now,
let's bring them all together. The process involves three main stages:

1.  **Build the Device Layer**: We'll create a `DeviceCoordinator` to manage our
    collection of `Device`s.
2.  **Build the Task Layer**: We'll instantiate our custom processors and producers and
    combine them in a `TaskLayer`. We will also add a stock
    `MaximumStepsTerminationChecker` to ensure our episodes eventually end.
3.  **Build the Environment**: We'll pass the Device Layer and Task Layer into the main
    `Environment` class.

The function below encapsulates this entire process.

### The Environment Builder Function

```python
import datetime
import numpy as np
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types

# Import REAF building blocks.
from reaf.common import device_sequence_coordinator
from reaf.common import maximum_steps_termination_checker
from reaf.common import time_trigger
from reaf.common import environment_reset_from_callable
from reaf.core import environment
from reaf.core import device_layer as dl_module
from reaf.core import task_layer as tl_module


# Import the components we built in the previous steps.
import .ideal_robot
import .ideal_camera
from .task_components import (
    ConstantFactorCommandsProcessor,
    RelativePositionFeaturesProducer,
)
from .robot_device import RobotDevice
from .camera_device import CameraDevice


def build_environment(
    robot: ideal_robot.Robot,
    left_camera: ideal_camera.RgbCamera,
    right_camera: ideal_camera.RgbCamera,
    max_duration: datetime.timedelta,
    control_timestep: datetime.timedelta,
) -> environment.Environment:
  """Builds our custom environment by assembling all the components."""

  # 1. Build the Device Layer from our devices and specified control timestep.
  robot_device = RobotDevice(robot)
  left_camera_device = CameraDevice(left_camera)
  right_camera_device = CameraDevice(right_camera)
  coordinator = device_sequence_coordinator.DeviceSequenceCoordinator(
      [robot_device, left_camera_device, right_camera_device],
      "device_coordinator",
  )

  device_layer = dl_module.DeviceLayer(
      device_coordinator=coordinator,
      commands_trigger=None,
      measurements_trigger=time_trigger.TimeTrigger(period=control_timestep),
  )

  # 2. Instantiate and build the Task Layer from our custom components.
  # First, define the agent's action spec.
  torque_action_spec = gdmr_types.UnboundedArraySpec(
    shape=(robot.dofs,), dtype=np.float32)

  # Instantiate the command processor to convert torque to current.
  torque_to_current_processor = ConstantFactorCommandsProcessor(
      produced_key=ROBOT_COMMAND_CURRENT_KEY,
      consumed_key=AGENT_TORQUE_KEY,
      consumed_spec=torque_action_spec,
      constant_factor=1.0 / ideal_robot.ROBOT_MOTOR_CONSTANT,
  )

  # Instantiate the features producer to compute relative position.
  robot_pos_spec = specs.Array(shape=(robot.dofs,), dtype=np.float32)
  relative_position_producer = RelativePositionFeaturesProducer(
      position_spec=robot_pos_spec
  )

  # Instantiate a step termination checker.
  termination_checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
      max_steps=max_duration // control_timestep
  )

  # Assemble the Task Layer.
  task_layer = tl_module.TaskLayer(
      commands_processors=[torque_to_current_processor],
      features_producers=[relative_position_producer],
      termination_checkers=[termination_checker],
  )

  # 3. Build the final Environment.
  # For now we do not specify any reset function.
  reset_fn = environment_reset_from_callable.EnvironmentResetFromCallable(
    lambda _: None
  )

  return environment.Environment(
      device_layer=device_layer,
      task_layer=task_layer,
      environment_reset=reset_fn,
  )
```

And that's it! We have now constructed a complete, customized REAF environment
from modular, reusable components. This explicit, layered approach is the core
strength of REAF, as it allows you to easily swap out any part of the system — a
`Device`, a `FeaturesProducer`, or a `RewardProvider` — to suit the needs of
your next experiment.

## ✅ Step 4: Putting It All Together

Congratulations! You've successfully built a custom REAF environment with a
modular Task Logic Layer. We now have a builder function,
`build_custom_environment`, that assembles all our components.

The final step is to run it. We can now reuse the same standard `main` function
from the first tutorial. We'll create our simulated robot, pass it to our new
builder function, create a simple agent, and start the `Runloop`.

### The Main Execution Block

Here is the complete code to run your new environment. This block brings
together the `Robot` simulation, our environment builder, a simple agent that
sends zero-torque commands, and the `StdoutLogger` we defined in the overview.

```python
from collections.abc import Sequence

from absl import app
import datetime
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
from gdm_robotics.runtime import runloop as runloop_lib
import numpy as np

import .ideal_robot
import .ideal_camera
import .environment_creator_with_task_layer
import .stdout_logger


class _ZeroAgent(gdmr_policy.Policy[np.ndarray]):
  """Agent that always returns zero action."""

  def __init__(
      self,
      action_key: str,
      num_dofs: int,
  ):
    self._action_key = action_key
    self._num_dofs = num_dofs

    # This policy has no hidden state. We create a dummy empty state.
    self._dummy_state = np.empty(0, dtype=np.float32)

  def initial_state(
      self,
  ) -> gdmr_types.StateStructure[np.ndarray]:
    """Returns the policy initial state."""
    return self._dummy_state

  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: gdmr_types.StateStructure[np.ndarray],
  ) -> tuple[
      tuple[
          gdmr_types.ActionType,
          gdmr_types.ExtraOutputStructure[np.ndarray],
      ],
      gdmr_types.StateStructure[np.ndarray],
  ]:

    action = {self._action_key: np.zeros(self._num_dofs, dtype=np.float32)}
    return (action, {}), self._dummy_state

  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    return (
        {
          self._action_key: gdmr_types.UnboundedArray(
            shape=self._num_dofs,
            dtype=np.float32,
          ),
        },
        {},
    ), specs.Array(shape=(), dtype=np.float32)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Create the robot and cameras devices.
  robot = ideal_robot.Robot(num_dofs=7)
  left_camera = ideal_camera.RgbCamera()
  right_camera = ideal_camera.RgbCamera()

  env = environment_creator_with_task_layer.build_environment(
    robot,
    left_camera,
    right_camera,
    datetime.timedelta(seconds=20),
    datetime.timedelta(milliseconds=50)
  )

  print(f"{env.observation_spec()=}")
  print(f"{env.action_spec()=}")

  policy = _ZeroAgent("current_reference", robot.num_dofs)

  runloop = runloop_lib.Runloop(
    env, policy, loggers=(stdout_logger.StdoutLogger()),
  )
  runloop.run(num_episodes=5)

  robot.shutdown()
  left_camera.shutdown()
  right_camera.shutdown()


if __name__ == "__main__":
  app.run(main)
```

### What to Expect

When you run this script, you will see the log output for the episodes. Notice
that the observation printed by the logger will be contain both the absolute and
the relative joint position.

You have now mastered the fundamentals of creating and customizing task logic in
REAF environments. You can use this modular structure as a starting point for
all your future robotics experiments!
