# Tutorials Overview

Welcome to the REAF tutorials! 🚀 This section will guide you through the process
of building robotics environments with REAF. We'll start with a simple "Hello,
World!" example and progressively introduce more advanced concepts and
customizations.

By the end of these tutorials, you will be able to build your own modular and
reusable environments.

********************************************************************************

## Prerequisites

To get the most out of these tutorials, you should be familiar with the
following concepts:

*   The basic agent-environment interaction loop as described in
    **[Reinforcement Learning: An Introduction (Chapter 3)](http://incompleteideas.net/book/RLbook2020.pdf)**.
*   The
    **[DeepMind Environment API (`dm_env`)](https://github.com/google-deepmind/dm_env/blob/master/docs/index.md)**,
    which defines the standard interface that REAF environments implement. Note
    that REAF implements the
    [GDM Robotics Environment interface](https://github.com/google-deepmind/gdm_robotics/blob/main/src/gdm_robotics/interfaces/environment.py)
    which is a more precise, but compatible, implementation of the `dm_env`.

********************************************************************************

## The Anatomy of a Tutorial

Each tutorial will focus on building a specific `Environment`. However, the way
we interact with the environment once it's built will be the same in every
example. All REAF environments are compliant with the standard GDM Robotics and
`dm_env` interfaces, so we can use a common set of tools to run them.

In each tutorial, we will use two main components to interact with our
environment:

1.  **The Run Loop**: A `gdm_robotics.runtime.runloop.Runloop` will manage the
    core interaction loop: it takes an agent's action, passes it to the
    environment, and receives the next timestep.
2.  **The Logger**: A simple `episodic_logger` will be used to print the results
    of each step to the console. This allows us to see what's happening whilst
    the environment as the agent interacts with each other.

### Our Standard Logger

To keep the focus on building environments, we will use the same simple logger
implementation in all tutorials. This logger prints the actions, observations,
and rewards for each step. Here is the code so you can familiarise yourself with
it:

```python
import logging
from collections.abc import Mapping
from typing import Any

import dm_env
from gdm_robotics.interfaces import episodic_logger
from gdm_robotics.interfaces import types as gdmr_types

class StdoutLogger(episodic_logger.EpisodicLogger):
  """A simple logger that prints episode interactions to the console."""

  def reset(self, timestep: dm_env.TimeStep) -> None:
    """Called at the beginning of a new episode."""
    logging.info("========= EPISODE RESET =========")
    logging.info("Initial Observations: %s", timestep.observation)

  def record_action_and_next_timestep(
      self,
      action: gdmr_types.ActionType,
      next_timestep: dm_env.TimeStep,
      policy_extra: Mapping[str, Any],
  ) -> None:
    """Called after each step to log the action and resulting timestep."""
    logging.info("--- Step ---")
    logging.info("Action: %s", action)
    logging.info("Observation: %s", next_timestep.observation)
    logging.info("Reward: %s", next_timestep.reward)
```
