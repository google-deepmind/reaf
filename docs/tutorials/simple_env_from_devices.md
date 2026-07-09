# Tutorial: Implementing a REAF Device

Welcome to your first hands-on REAF tutorial! 🤖 Implementing a **`Device`** is
the first step to connecting any new hardware or simulation to the REAF
ecosystem.

In this guide, we will: 1. Introduce two simple Python classes that simulate a
robot's hardware and a camera API. 2. Show you how to wrap these APIs within a
REAF **`Device`**. 3. Build the simplest possible environment to test our new
`Device`, which will directly expose the robot's measurements and commands.

--------------------------------------------------------------------------------

## Step 1: Understanding the Hardware API

Before we can write any REAF code, we first need to understand the Application
Programming Interface (API) of the hardware we want to control. In a real-world
scenario, this could be a set of ROS2 topics, a vendor-specific SDK, or a direct
connection over a data bus.

For this tutorial, we don't need a physical robot. Instead, we'll use a simple
Python class that simulates a robot arm. This allows us to focus on the REAF
concepts without getting bogged down by hardware-specific details.

### Our Simulated Robot API

Let's imagine our robot arm provides a Python API with the following
capabilities:

*   **`get_state()`**: Returns the current joint positions and velocities.
*   **`set_currents()`**: Sends motor current commands to the joints.
*   **`reset_state()`**: Resets the robot to a specified joint configuration.

The code below provides a simple implementation of this API. **You don't need to
understand the implementation details (like the threading or physics
integration)**; just focus on the public methods, as these are what our REAF
`Device` will interact with.

```python
import threading
import time
import numpy as np

ROBOT_MOTOR_CONSTANT = 0.24  # Nm/A
_DT = 0.01

class Robot:
  """
  A simulated robot arm that mimics a real hardware API.
  It runs its own internal physics simulation in a separate thread.
  """

  def __init__(self, num_dofs: int):
    """Initializes the robot with a given number of degrees of freedom."""
    self._dofs = num_dofs
    self._measurements_mutex = threading.Lock()
    self._position = np.zeros(num_dofs)
    self._velocity = np.zeros(num_dofs)
    self._command_mutex = threading.Lock()
    self._current_reference = np.zeros(num_dofs)
    self._should_stop = False
    self._thread = threading.Thread(target=self._integrator)
    self._thread.start()

  @property
  def dofs(self) -> int:
    """The number of degrees of freedom of the robot arm."""
    return self._dofs

  def shutdown(self) -> None:
    """Safely shuts down the background simulation thread."""
    self._should_stop = True
    self._thread.join()

  def reset_state(
      self, new_position: np.ndarray, new_velocity: np.ndarray | None = None
  ) -> None:
    """Resets the robot's joint positions and velocities."""
    with self._command_mutex, self._measurements_mutex:
      self._current_reference = np.zeros(self._dofs)
      self._position = new_position
      self._velocity = new_velocity if new_velocity is not None else np.zeros(self._dofs)

  def set_currents(self, currents: np.ndarray) -> None:
    """Sets the target motor currents for the robot's joints."""
    with self._command_mutex:
      self._current_reference = currents

  def get_state(self) -> tuple[np.ndarray, np.ndarray]:
    """Gets the latest joint positions and velocities from the robot."""
    with self._measurements_mutex:
      return self._position.copy(), self._velocity.copy()

  def _integrator(self) -> None:
    """Integrates the system dynamics in a background thread."""
    last_update_time = time.time()
    while not self._should_stop:
      with self._command_mutex:
        currents = self._current_reference
      with self._measurements_mutex:
        self._position += _DT * self._velocity
        self._velocity += _DT * currents
      last_update_time += _DT
      time.sleep(max(0, last_update_time - time.time()))
```

### Our Simulated Camera API

Similar to the robot arm, we can also simulate a camera. This `RgbCamera` class
will provide a simple API to get image frames.

*   **`get_frame()`**: Returns the latest RGB image as a NumPy array.

The code below provides a simple implementation of this API, generating random
frames in a background thread.

```python
class RgbCamera:
  """A simulated RGB camera that provides image frames."""

  def __init__(self, width: int = 640, height: int = 480, refresh_rate: int = 30):
    """Initializes the camera with a given frame size."""
    self._width = width
    self._height = height
    self._frame_mutex = threading.Lock()
    self._frame = np.zeros((height, width, 3), dtype=np.uint8)
    self._should_stop = False
    self._refresh_rate = 1.0 / refresh_rate
    self._thread = threading.Thread(target=self._frame_generator)
    self._thread.start()

  @property
  def width(self) -> int:
    return self._width

  @property
  def height(self) -> int:
    return self._height

  def shutdown(self) -> None:
    """Safely shuts down the background frame generation thread."""
    self._should_stop = True
    self._thread.join()

  def get_frame(self) -> np.ndarray:
    """Gets the latest RGB image frame from the camera."""
    with self._frame_mutex:
      return self._frame.copy()

  def _frame_generator(self) -> None:
    """Generates frames in a background thread."""
    last_update_time = time.time()
    while not self._should_stop:
      # Simulate generating a new frame
      with self._frame_mutex:
        # Example: Generate a frame with random noise
        self._frame = np.random.randint(0, 256, size=(self._height, self._width, 3), dtype=np.uint8)
      last_update_time += self._refresh_rate
      time.sleep(max(0, last_update_time - time.time()))
```

## Step 2: Build the REAF Devices

Now that we have a simulated robot API, let's build our REAF **`Device`s**. A
`Device` is a Python class that acts as a wrapper, adapting a specific hardware
API to the standardized REAF interface.

Think of a `Device` as a "driver" for a piece of hardware. Ideally, each
`Device` should manage the most minimal and independent component possible. For
example, we'll create one `Device` for our robot arm and a separate one for a
camera.

A `Device` has two primary responsibilities:

*   **Define its "contract"**: It must describe the data it produces
    (measurements) and consumes (commands). This is done by implementing the
    `measurements_spec()` and `commands_spec()` methods.
*   **Implement the communication**: It must contain the actual code to read
    from the hardware's sensors (`get_measurements()`) and send signals to its
    actuators (`set_commands()`).

--------------------------------------------------------------------------------

### Implementing the RobotDevice

Let's create our `RobotDevice`. It's a good practice to define the dictionary
keys that we'll use for commands and measurements as constants. This prevents
typos and makes the code easier to read and maintain.

The entire implementation, including the specs and the methods that call our
`Robot` API, is shown in the class below.

```python
from collections.abc import Mapping
import numpy as np
from typing_extensions import override

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import device

import .ideal_robot

# Define constants for the dictionary keys to avoid typos.
ROBOT_MEASUREMENT_POSITION_KEY = "joint_position"
ROBOT_MEASUREMENT_VELOCITY_KEY = "joint_velocity"
ROBOT_COMMAND_CURRENT_KEY = "current_reference"

class RobotDevice(device.Device):
  """A REAF Device that wraps our simulated ideal_robot.Robot."""

  def __init__(self, robot: ideal_robot.Robot):
    """Initializes the device with an instance of the robot API."""
    self._robot = robot

  # Part 1: Define the "contract" with specs.

  @override
  def measurements_spec(self) -> dict[str, specs.Array]:
    """Describes the data this Device provides."""
    return {
        ROBOT_MEASUREMENT_POSITION_KEY: specs.Array(
            shape=(self._robot.dofs,), dtype=np.float32
        ),
        ROBOT_MEASUREMENT_VELOCITY_KEY: specs.Array(
            shape=(self._robot.dofs,), dtype=np.float32
        ),
    }

  @override
  def commands_spec(self) -> dict[str, gdmr_types.AnyArraySpec]:
    """Describes the data this Device accepts."""
    return {
        ROBOT_COMMAND_CURRENT_KEY: gdmr_types.UnboundedArray(
            shape=(self._robot.dofs,),
            dtype=np.float32,
        )
    }

  # Part 2: Implement the communication with the hardware API.

  @override
  def get_measurements(self) -> dict[str, np.ndarray]:
    """Calls the robot API to get sensor data."""
    position, velocity = self._robot.get_state()
    # The returned dictionary keys must match the measurements_spec.
    return {
        ROBOT_MEASUREMENT_POSITION_KEY: position,
        ROBOT_MEASUREMENT_VELOCITY_KEY: velocity,
    }

  @override
  def set_commands(self, commands: Mapping[str, np.ndarray]) -> None:
    """Calls the robot API to send actuator commands."""
    # The incoming dictionary keys will match the commands_spec.
    currents = commands[ROBOT_COMMAND_CURRENT_KEY]
    self._robot.set_currents(currents)
```

### Implementing the CameraDevice

Now, let's implement the CameraDevice. Similar to the `RobotDevice`, this class
will wrap our `RgbCamera` API, exposing its functionality through the REAF
Device interface.

We'll define a constant for the key used to represent the camera frame in the
measurements dictionary. Additionally, we'll add an optional name parameter to
the `__init__` method. If provided, this name will be used as a prefix for the
measurement keys, which is useful when multiple devices of the same type are
used in an environment.

The implementation details are shown below:

```py
from collections.abc import Mapping
import numpy as np
from typing_extensions import override

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import device

import .ideal_camera

# Define constants for the dictionary keys.
CAMERA_MEASUREMENT_FRAME_KEY = "rgb"

class CameraDevice(device.Device):
  """A REAF Device that wraps our simulated ideal_camera.RgbCamera."""

  def __init__(self, camera: ideal_camera.RgbCamera, name: str | None = None):
    """Initializes the device with an instance of the camera API.

    Args:
      camera: An instance of the simulated RgbCamera.
      name: An optional name for this device. If provided, it will be used
        as a prefix for the measurement keys (e.g., "my_camera/rgb").
    """
    self._camera = camera
    self._name = name

  def _prefixed_key(self, key: str) -> str:
    """Returns a key, prefixed with the device name if available."""
    return f"{self._name}/{key}" if self._name else key

  # Part 1: Define the "contract" with specs.

  @override
  def measurements_spec(self) -> dict[str, specs.Array]:
    """Describes the data this Device provides.

    The key for the RGB frame is prefixed with the device name if one was
    provided during initialization.
    """
    return {
        self._prefixed_key(CAMERA_MEASUREMENT_FRAME_KEY): specs.Array(
            shape=(self._camera.height, self._camera.width, 3), dtype=np.uint8
        ),
    }

  @override
  def commands_spec(self) -> dict[str, gdmr_types.AnyArraySpec]:
    """Describes the data this Device accepts.

    Our simulated camera does not accept any commands, so this returns an
    empty dictionary.
    """
    return {}

  # Part 2: Implement the communication with the hardware API.

  @override
  def get_measurements(self) -> dict[str, np.ndarray]:
    """Calls the camera API to get sensor data.

    The returned dictionary contains the latest RGB frame, with a key that
    matches the one defined in `measurements_spec`.
    """
    frame = self._camera.get_frame()
    return {
        self._prefixed_key(CAMERA_MEASUREMENT_FRAME_KEY): frame,
    }

  @override
  def set_commands(self, commands: Mapping[str, np.ndarray]) -> None:
    """Calls the camera API to send actuator commands.

    Since `commands_spec` is empty, this method does nothing.
    """
    # No commands are accepted by this device.
    pass
```

## Step 3. Create the environment

With our `Device` implementations complete, we can now create the environment.
In this introductory tutorial, we'll build the simplest possible environment:
one that directly exposes all measurements and commands from our devices,
without any custom task logic.

```py
import datetime
from reaf.common import environment_from_devices
from reaf.core import environment

import .camera_device
import .ideal_robot
import .ideal_camera
import .robot_device

# Our environment steps at 20Hz.
_ENV_DT = datetime.timedelta(milliseconds=50)


def create_environment(
  robot: ideal_robot.Robot,
  left_camera: ideal_camera.RgbCamera,
  right_camera: ideal_camera.RgbCamera,
  episode_duration: datetime.timedelta) -> environment.Environment:

  robot_device = robot_device.RobotDevice(robot)
  left_camera_device = camera_device.CameraDevice(
    left_camera, name="left_camera")
  right_camera_device = camera_device.CameraDevice(
    right_camera, name="right_camera")

  return environment_from_devices.environment_from_devices(
    devices=(robot_device, left_camera, right_camera),
    control_timestep=_ENV_DT,
    episode_max_duration=episode_duration,
  )
```

## Step 4. Run the environment!

This step demonstrates how to run the environment created in Step 3. We'll use a
simple "zero agent" that always sends zero commands to the robot. This allows us
to observe the environment's measurements without any complex control logic.

The provided Python script imports necessary libraries, defines a `_ZeroAgent`
policy that always outputs zero commands, and in the `main` function: 1.
Instantiates the simulated `Robot` and `RgbCamera` clients. 2. Creates the REAF
environment using the function we created before. 3. Prints the environment's
observation and action specifications. 4. Sets up a `runloop_lib.Runloop` with
the environment, the `_ZeroAgent`, and a `stdout_logger.StdoutLogger`. 5. Runs
the environment for 5 episodes. This will be roughly 100 seconds (20 seconds per
episode). 6. Shuts down the robot and camera instances. This setup allows you to
see the environment in action, with measurements being generated by the devices
and logged to the console, even without a sophisticated control policy.

```py
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
import .environment_creator
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

  env = environment_creator.create_environment(
    robot, left_camera, right_camera, datetime.timedelta(seconds=20)
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
