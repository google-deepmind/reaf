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
"""Episode logger that logs using absl logging."""

from collections.abc import Mapping
from typing import Any

from absl import logging
import dm_env
from gdm_robotics.interfaces import episodic_logger as gdmr_logger
from gdm_robotics.interfaces import types as gdmr_types


class StdoutLogger(gdmr_logger.EpisodicLogger):
  """Episode logger that logs using absl logging."""

  def reset(self, timestep: dm_env.TimeStep) -> None:
    logging.info("Episode reset. Observations: %s", timestep.observation)

  def record_action_and_next_timestep(
      self,
      action: gdmr_types.ActionType,
      next_timestep: dm_env.TimeStep,
      policy_extra: Mapping[str, Any],
  ) -> None:
    """Logs an action and the resulting timestep."""
    logging.info("Step")
    logging.info("Action: %s", action)
    logging.info("Next state observations: %s", next_timestep.observation)
    logging.info("Reward: %s", next_timestep.reward)

  def write(self) -> None:
    """Writes the current episode logged data."""
    # Nothing to do.
    pass
