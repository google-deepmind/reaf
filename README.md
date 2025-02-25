# Robotics Environment Authoring Framework (REAF)

The Robotics Environment Authoring Framework (REAF) simplifies creating
environments that adhere to the
[GDM Robotics Environment interface](https://github.com/google-deepmind/gdm_robotics/interfaces/environment.py).

## How to install

`reaf` can be installed from PyPI using `pip`:

```bash
pip install reaf
```

## Directory Structure

Currently, the directory structure is designed as standalone subdirectories with
a well-defined dependency graph.

```
reaf
├── core: Core libraries and interfaces for REAF.
├── testing: General tooling for testing REAF interfaces and environments.
├── common: Libraries with shared functionality across setups and platforms.
```

## Design

REAF is a framework designed to simplify the creation of robotics environments.
It adopts a layered architecture to promote modularity and reusability. The core
components of a REAF environment are:

1.  **Environment:** The top-level interface for interacting with the
    environment, conforming to the
    [GDM Robotics Environment interface](https://github.com/google-deepmind/gdm_robotics/interfaces/environment.py).
    It handles stepping, resetting, and action/observation specs.
2.  **Task Logic Layer (TLL):** Responsible for defining the task itself,
    including reward calculation, termination conditions, features generation,
    and commands processing.
3.  **Data Acquisition and Control Layer (DACL):** Interfaces with the physical
    or simulated robotic setup, managing commands to actuators and retrieving
    measurements from sensors.
4.  **Adapters:** Bridge the gap between the abstract GDMR interfaces and the
    specific requirements of the TLL. These adapters translate agent actions
    into TLL commands and TLL features into agent observations.
5.  **Reset and End of Episode Handlers:** Support customized behavior during
    environment resets and episode termination.

### Environment

The `Environment` class serves as the primary interface for interacting with the
robotic environment. It coordinates the interactions between the TLL and DACL,
manages the environment's state, and handles stepping through the environment.
Key functionalities include:

*   **`reset_with_options()`:** Resets the environment to a new initial state
    based on the provided options. This involves resetting the DACL, computing
    initial features and observations.
*   **`step()`:** Advances the environment by one step. This method takes an
    agent action, processes it into commands, steps the DACL, computes features,
    reward, discount, termination conditions, and new observations, and returns
    a `TimeStep` object containing this information.
*   **`action_spec()`:** Returns the specification for valid agent actions. This
    is determined by the `ActionSpaceAdapter`.
*   **`timestep_spec()`:** Returns the specification for the `TimeStep` objects
    returned by `step()` and `reset()`.
*   **Logging:** Facilitates adding and removing loggers to monitor internal
    operations.

### Task Logic Layer (TLL)

The TLL defines the logic and rules governing the robotic task. It comprises
several core components:

*   **`FeaturesProducer`:** Generates additional features based on existing
    features and measurements from the DACL. Each producer has a
    `produced_features_spec()` defining the features it generates and
    `required_features_keys()` indicating the features it depends on.
*   **`CommandsProcessor`:** Modifies commands before they are sent to the DACL.
    Processors can transform, filter, or augment commands. The
    `consumed_commands_spec()` describes the commands accepted by the processor,
    and `produced_commands_keys()` defines the output commands.
*   **`RewardProvider`:** Calculates the reward signal based on the current
    features. It exposes a `reward_spec()` defining the structure of the reward.
*   **`TerminationChecker`:** Determines whether the episode should terminate
    based on features and returns a `TerminationResult` indicating the type of
    termination.
*   **`DiscountProvider`:** Computes the discount factor based on features and
    the termination state.
*   **`FeaturesObserver`:** Passive components that observe features without
    modifying them. This is useful for logging or analysis.
*   **`Logger`:** Records measurements, features, and commands during
    environment interactions. Methods like `record_measurements()`,
    `record_features()`, and `record_commands_processing()` are called at
    specific points in the environment's lifecycle.

The TLL also provides methods to:

*   **`compute_all_features()`:** Computes all features based on measurements
    from the DACL and the output of `FeaturesProducer`s.
*   **`compute_final_commands()`:** Processes the policy's commands using the
    `CommandsProcessor`s and outputs the DACL command.
*   **`compute_reward()`:** Calculates the reward using the `RewardProvider`.
*   **`check_for_termination()`:** Checks termination conditions using
    `TerminationChecker`s.
*   **`compute_discount()`:** Computes the discount using the
    `DiscountProvider`.
*   **`validate_spec()`:** Verifies the consistency of the specs across the TLL
    and DACL.

### Data Acquisition and Control Layer (DACL)

The DACL serves as the bridge between the REAF environment and the robotic
hardware. It's responsible for sending commands to the robot and receiving
measurements from sensors. The DACL is built around:

*   **`Device`:** Represents a single hardware component (e.g., robot arm,
    camera). It provides methods like `set_commands()` and `get_measurements()`
    for interacting with the hardware.
*   **`DeviceCoordinator`:** Manages a collection of `Device` objects,
    coordinating their actions and data exchange. It provides lifecycle
    management through `start()` and `stop()` methods and synchronization points
    through `before_set_commands()` and `before_get_measurements()`.

The DACL's key functions are:

*   **`begin_stepping()`:** Initializes the DACL and returns the initial
    measurements.
*   **`step()`:** Sends commands to the devices, retrieves new measurements, and
    returns them.
*   **`end_stepping()`:** Performs cleanup operations at the end of an episode.
*   **`commands_spec()`:** Returns the specification for valid commands. The
    user is expected to pass the full command dictionary.
*   **`measurements_spec()`:** Returns the specification for the measurements
    returned by `get_measurements()`.

### Adapters

REAF utilizes adapters to translate between the generic agent interface and the
specific format required by the TLL.

*   **`ActionSpaceAdapter`:** Converts the agent's actions into a commands
    dictionary understood by the TLL.
*   **`ObservationSpaceAdapter`:** Transforms the features generated by the TLL
    into observations suitable for the agent.

### Reset and End of Episode Handler

*   **`EnvironmentReset`:** Defines the reset behavior of the environment,
    including a `do_reset()` method and a default reset configuration. The reset
    is a complex step and very hard to predict how this will be written. The
    intention is to leave complete control to the user so they can pass whatever
    object is needed to the reset without any restrictions. The user can use
    this to check that everything is working properly, reset the state of any
    features producers, etc...
*   **`EndOfEpisodeHandler`:** Provides a callback function,
    `on_end_of_episode_stepping()`, that is invoked at the end of each episode
    after the last step. This is useful for logging, cleanup, or custom logic
    that needs to be executed when an episode ends.

## Licence and Disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing 
permissions and limitations under those licenses.

This is not an official Google product.
