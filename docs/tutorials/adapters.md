# Tutorial: Adapting to the Agent with Adapters

In our previous tutorials, we focused on building the internal logic of a REAF
environment. Now, we'll cover the crucial final layer: the **Adapters**. These
components act as the bridge between your structured REAF environment and the
outside world of the agent.

--------------------------------------------------------------------------------

## The Core Problem: Structure vs. Generality

The standard `dm_env` interface, which REAF adheres to, is intentionally
generic. An agent's **action** and the environment's **observation** are both
flexible `tree.Structure` objects (like nested dictionaries or lists of
tensors). This allows a single agent to interface with many different kinds of
environments.

However, inside a REAF environment, we need more **structure**. The Task Layer and Device Layer
rely on named, structured dictionaries (`Mapping[str, np.ndarray]`) to
automatically validate data flow, chain processors, and connect to devices.

**Adapters solve this mismatch.** They are translators that convert data between
the generic format an agent uses and the structured format REAF uses internally.

--------------------------------------------------------------------------------

## `ActionSpaceAdapter`: From Flat Action to Structured Command

The **`ActionSpaceAdapter`** translates the agent's action into the structured
command dictionary that the Task Layer expects.

*   **Direction**: Agent → Environment
*   **Common Use Case**: A general-purpose RL agent often outputs its action as
    a simple, flat `np.ndarray` (e.g., `[0.1, -0.2, 0.5, ...]`). However, our
    Task Layer needs a named dictionary to know what this action means (e.g.,
    `{'agent_torque': [0.1, -0.2, 0.5, ...]}`).

The adapter performs this mapping. It takes the agent's simple array and wraps
it in the dictionary structure the Task Layer requires.

**Example Implementation:** Let's create an adapter that takes a flat array from
the agent and maps it to the `AGENT_TORQUE_KEY` our Task Layer consumes. As this use
case is very common, REAF already provides an adapter for this
`reaf.common.partitioner_action_space_adapter`.

```py
from reaf.common import partitioner_action_space_adapter

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

  # 3. Before building the final Environment create the action space adapter.

  action_spec = gdmr_types.UnboundedArraySpec(
    shape=(robot.dofs, ), dtype=np.float32)

  # Define the partitions, i.e. how to map from the flat array to a dictionary.
  # In this example we have a single partition.
  torques_partition = partitioner_action_space_adapter.PartitionInfo(
    command_key=AGENT_TORQUE_KEY,
    start_index=0,
    length=robot.dofs
  )
  action_space_adapter = partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
    partitions=(torques_partition, ),
    action_spec=action_spec,
  )

```

## `ObservationSpaceAdapter`: From Rich Features to Filtered Observation

The **`ObservationSpaceAdapter`** does the reverse. It takes the rich, internal
dictionary of **features** and filters or formats it into the specific
**observation** the agent should receive.

*   **Direction**: Environment → Agent
*   **Common Use Case**: The Task Layer might compute many features. Some are for the
    agent, but others might be **privileged information** used only for
    calculating rewards or logging diagnostics (e.g., the true position of a
    hidden goal). We don't want to leak this privileged information to the agent
    in its observation.

**Example Implementation:** Let's create an adapter that exposes the
`relative_joint_position` and `joint_velocity` to the agent, but hides the
(absolute) `joint_position`. As this is a very common use case, REAF already
provides an `ObservationSpaceAdapter` that filters features, in
`reaf.core.default_observation_space_adapter`.

```python
from reaf.core import default_observation_space_adapter

# Keys for the features we want the agent to see.
OBSERVED_KEYS = ["relative_joint_position", "joint_velocity"]

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

  # 3. Before building the final Environment create the observation space
  # adapter.

  # We get all the specs from the Task Layer.
  task_features_specs = task_layer.features_spec(device_layer.measurements_spec())
  observation_space_adapter = (
    default_observation_space_adapter.DefaultObservationSpaceAdapter(
      task_features_spec=task_features_specs,
      selected_features=OBSERVED_KEYS
    )
  )

```

By cleanly separating the agent's interface from the environment's internal
logic, adapters make your code more modular, reusable, and easier to debug.

## ✅ Final Step: Integrating the Final Adapters

We've now designed our custom adapters. The final step is to integrate them into
our environment builder function. This is where we see the power of REAF's
modularity: we can completely change the public-facing "API" of our environment
for the agent without touching any of the internal Task Layer or Device Layer logic we've
already built.

### Updating the Environment Builder

We will now modify our `build_custom_environment` function one last time. We
will use the adapters we created in the previous sections in the Environment
initializer.

Here is the complete, updated builder function:

```python
import numpy as np

# ... (all previous imports remain the same)
from reaf.common import partitioner_action_space_adapter
from reaf.core import default_observation_space_adapter
from reaf.core import device_layer as dl_module
from reaf.core import task_layer as tl_module
# ...

# Key for the agent's action. This is internal to the TLL.
AGENT_TORQUE_KEY = "agent_torque"


def build_custom_environment(
    robot: ideal_robot.Robot,
    left_camera: ideal_camera.RgbCamera,
    right_camera: ideal_camera.RgbCamera,
    max_duration: datetime.timedelta,
    control_timestep: datetime.timedelta,
) -> environment.Environment:
  """Builds the final environment with custom, agent-friendly adapters."""

  # --- Steps 1 & 2: Build Device Layer and Task Layer (Identical to previous tutorial) ---
  robot_device = robot_device.RobotDevice(robot)
  left_camera_device = camera_device.CameraDevice(
      left_camera, name="left_camera"
  )
  right_camera_device = camera_device.CameraDevice(
      right_camera, name="right_camera"
  )

  coordinator = device_sequence_coordinator.DeviceSequenceCoordinator(
      [robot_device, left_camera_device, right_camera_device],
      "device_coordinator",
  )

  device_layer = dl_module.DeviceLayer(
      device_coordinator=coordinator,
      commands_trigger=None,
      measurements_trigger=time_trigger.TimeTrigger(period=control_timestep),
  )

  # Instantiate the command processor to convert torque to current.
  torque_to_current_processor = ConstantFactorCommandsProcessor(
      # ... (same as before)
  )

  relative_position_producer = RelativePositionFeaturesProducer(
      # ... (same as before)
  )

  termination_checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
      max_steps=max_duration // control_timestep
  )

  task_layer = tl_module.TaskLayer(
      commands_processors=[torque_to_current_processor],
      features_producers=[relative_position_producer],
      termination_checkers=[termination_checker],
  )

  # --- Step 3: Build Environment with NEW Adapters ---

  # The agent now sees a simple, flat array for its action space.
  action_spec = gdmr_types.UnboundedArraySpec(
    shape=(robot.dofs, ), dtype=np.float32)

  torques_partition = partitioner_action_space_adapter.PartitionInfo(
    command_key=AGENT_TORQUE_KEY,
    start_index=0,
    length=robot.dofs
  )
  action_space_adapter = partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
    partitions=(torques_partition, ),
    action_spec=action_spec,
  )

  # The agent will only observe these two features.
  # 'joint_position' is now hidden from the agent.
  observed_keys = ["relative_joint_position", "joint_velocity"]
  task_features_specs = task_layer.features_spec(
      device_layer.measurements_spec()
  )
  observation_space_adapter = (
      default_observation_space_adapter.DefaultObservationSpaceAdapter(
          task_features_spec=task_features_specs, selected_features=observed_keys
      )
  )

  # The reset handler is the same as before.
  reset_handler = RandomizedRobotReset(
      # ... (same as before)
  )


  # Pass the reset handler to the environment's constructor.
  return environment.Environment(
      device_layer=device_layer,
      task_layer=task_layer,
      environment_reset=reset_handler,
      action_space_adapter=action_space_adapter,
      observation_space_adapter=observation_space_adapter,
  )
```

### What to Expect

If you now run this new environment with the same `main` function as before, you
will notice two key differences:

1.  **New Action Spec**: If you were to print `env.action_spec()`, it would now
    show a simple `UnboundedArraySpec`, not a dictionary. Your agent can now
    send a flat NumPy array as its action, making it compatible with a wider
    range of standard RL algorithms.

2.  **Filtered Observations**: The `StdoutLogger` will now print observations
    that are dictionaries containing only `relative_joint_position` and
    `joint_velocity`. The `joint_position` is no longer part of the observation
    sent back to the agent, successfully hiding this privileged information.

You have now completed the full workflow of building a sophisticated,
customized, and agent-friendly robotics environment using REAF's modular
components!
