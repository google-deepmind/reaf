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
"""Proportional-derivative policy with constant setpoint."""

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np


class PdAgent(gdmr_policy.Policy[np.ndarray]):
  """Implements a PD controller for controlling to a constant setpoint."""

  def __init__(
      self,
      *,
      position_key: str,
      velocity_key: str,
      position_reference_key: str,
      p_diag_gains: np.ndarray,
      d_diag_gains: np.ndarray,
  ):
    self._position_key = position_key
    self._velocity_key = velocity_key
    self._position_reference_key = position_reference_key

    self._action_size = p_diag_gains.shape

    if d_diag_gains.shape != p_diag_gains.shape:
      raise ValueError(
          f"Gains do not match in size. P-gain: {p_diag_gains.shape}, D-gain:"
          f" {d_diag_gains.shape}"
      )

    self._p_gains = np.diag(p_diag_gains)
    self._d_gains = np.diag(d_diag_gains)

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
    """Takes a step with the policy given an environment timestep.

    Args:
      timestep: An instance of environment `TimeStep`.
      prev_state: The state of the policy at the time of calling `step`.

    Returns:
      A tuple of ((action, extra), state) with `action` indicating the action to
      be executed, `extra` policy-specific auxiliary information and state the
      policy state that will need to be provided to the next call to `step`.
    """
    reference = timestep.observation[self._position_reference_key]
    # Get the latest measurement needed to compute the control.
    position = timestep.observation[self._position_key]
    velocity = timestep.observation[self._velocity_key]

    action = np.asarray(
        self._p_gains @ (reference - position) - self._d_gains @ velocity
    ).astype(np.float32)

    return (action, {}), self._dummy_state

  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    return (
        specs.BoundedArray(
            shape=self._action_size,
            dtype=np.float32,
            minimum=-np.inf * np.ones(self._action_size),
            maximum=np.inf * np.ones(self._action_size),
        ),
        {},
    ), specs.Array(shape=(), dtype=np.float32)
