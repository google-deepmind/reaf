# Copyright 2026 Google LLC
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
"""Standalone utility for computing joint velocity commands.

This module provides the compute_velocity_command function which
implements a proportional controller for converting desired positions into
velocity commands.
"""

import numpy as np


def compute_velocity_command(
    desired_positions: np.ndarray,
    current_positions: np.ndarray,
    current_velocity_command: np.ndarray,
    minimum_velocity_command: np.ndarray,
    maximum_velocity_command: np.ndarray,
    feedback_gains: np.ndarray,
    max_acceleration: np.ndarray,
    control_timestep: float,
) -> np.ndarray:
  """Computes a velocity using a P-controller on the position error."""
  desired_velocity = feedback_gains * (desired_positions - current_positions)
  desired_delta_velocity = desired_velocity - current_velocity_command

  max_allowed_velocity_delta = max_acceleration * control_timestep
  clipped_desired_delta_velocity = np.clip(
      desired_delta_velocity,
      -max_allowed_velocity_delta,
      max_allowed_velocity_delta,
  )
  clipped_desired_velocity = (
      current_velocity_command + clipped_desired_delta_velocity
  )
  return np.clip(
      clipped_desired_velocity,
      minimum_velocity_command,
      maximum_velocity_command,
  )
