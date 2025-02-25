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
"""REAF termination checker for workspace limits."""

from collections.abc import Mapping
import dataclasses
from absl import logging
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import termination_checker


@dataclasses.dataclass(frozen=True, kw_only=True)
class Workspace:
  """Configuration for the physical workspace."""

  max_x: float
  max_y: float
  max_z: float
  min_x: float
  min_y: float
  min_z: float
  center = (0.0, 0.0, 0.0)


def create_cubic_workspace(size_meters: float) -> Workspace:
  return Workspace(
      max_x=size_meters / 2.0,
      max_y=size_meters / 2.0,
      max_z=size_meters / 2.0,
      min_x=-size_meters / 2.0,
      min_y=-size_meters / 2.0,
      min_z=-size_meters / 2.0,
  )


class WorkspaceLimitsTerminationChecker(termination_checker.TerminationChecker):
  """A class for checking if the robot goes outside of the workspace limits."""

  def __init__(
      self,
      name: str,
      workspace: Workspace,
      tcp_position_key: str,
      added_x_tolerance: float = 0,
      added_y_tolerance: float = 0,
      added_z_tolerance: float = 0,
  ):
    """A class for checking if the robot goes outside of the workspace limits.

    Args:
      name: Name of the workspace limits termination checker.
      workspace: The allowable workspace (bounds/limits) for the robot.
      tcp_position_key: The spec key for the position of the robot.
      added_x_tolerance: The added tolerance to the workspace bounds in the x
        axis (both positive and negative).
      added_y_tolerance: The added tolerance to the workspace bounds in the y
        axis (both positive and negative).
      added_z_tolerance: The added tolerance to the workspace bounds in the z
        axis (both positive and negative).
    """
    self._name = name
    self._workspace = workspace
    self._tcp_position_key = tcp_position_key
    self._added_x_tolerance = added_x_tolerance
    self._added_y_tolerance = added_y_tolerance
    self._added_z_tolerance = added_z_tolerance

  @property
  def name(self) -> str:
    return self._name

  def check_termination(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> termination_checker.TerminationResult:
    """Checks if the episode should terminate based on the workspace limits."""
    # tcp = tool center point
    tcp_position = required_features[self._tcp_position_key]
    if np.all(
        tcp_position
        > np.asarray([
            self._workspace.min_x - self._added_x_tolerance,
            self._workspace.min_y - self._added_y_tolerance,
            self._workspace.min_z - self._added_z_tolerance,
        ])
    ) and np.all(
        tcp_position
        < np.asarray([
            self._workspace.max_x + self._added_x_tolerance,
            self._workspace.max_y + self._added_y_tolerance,
            self._workspace.max_z + self._added_z_tolerance,
        ])
    ):
      return termination_checker.TerminationResult.DO_NOT_TERMINATE

    logging.info(
        'Terminating due to workspace limit violation. \n'
        'tcp_site_position (xyz): %s\n'
        'workspace: %s\n',
        tcp_position,
        self._workspace,
    )
    return termination_checker.TerminationResult.TERMINATE

  def required_features_keys(self) -> set[str]:
    """Returns the feature keys that are required to check the termination."""
    return set([self._tcp_position_key])
