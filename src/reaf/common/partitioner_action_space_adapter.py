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
"""Converts a flat tensor action into a dictionary of tensors."""

from collections.abc import Iterable, Mapping
import dataclasses

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import action_space_adapter


@dataclasses.dataclass(frozen=True, kw_only=True)
class PartitionInfo:
  command_key: str
  start_index: int = 0
  length: int | None = None  # If None, all the remaining elements are used.


class PartitionerActionSpaceAdapter(action_space_adapter.ActionSpaceAdapter):
  """Partitions a flat tensor action into a dictionary of tensors.

  Note that this class assumes the input action being a 1D array. It performs a
  basic range check at construction but no further checks on the compatibility
  between the action provided as input to `commands_from_environment_action` and
  the output commands.

  Objects of this class can be used to convert from a flat tensor, e.g. a numpy
  array, into a dictionary of tensors. In the simplest case, where no
  partitioning is required, objects of this class can be used to provide a
  dictionary-like structure to be used as commands as required by the REAF task
  logic layer. For example:

  1) Map from single tensor into a dictionary with a single entry.
  # Action (input): np.array([1., 2., 3., 4.])
  # Commands (output): dict("my_command": np.array([1., 2., 3., 4.]))

  action_spec = # Defined by the user.
  partitions = (
      PartitionInfo(command_key="my_command"),
  )

  adapter = PartitionerActionSpaceAdapter(
      partitions=partitions,
      action_spec=action_spec,
  )

  2) Split the single input tensor in multiple commands.
  # Action (input): np.array([1., 2., 3., 4., 5.])
  # Commands (output): dict(
  #       "command1": np.array([1., 2.]),
  #       "command2": np.array([3.]),
  #       "command3": np.array([4., 5.]),
  #       )

  action_spec = # Defined by the user.
  partitions = (
      PartitionInfo(command_key="command1", start_index=0, length=2),
      PartitionInfo(command_key="command2", start_index=2, length=1),
      PartitionInfo(command_key="command3", start_index=3, length=2),
  )

  adapter = PartitionerActionSpaceAdapter(
      partitions=partitions,
      action_spec=action_spec,
  )
  """

  def __init__(
      self,
      *,
      partitions: Iterable[PartitionInfo],
      action_spec: specs.BoundedArray,
  ):
    """Initializes the action space adapter.

    Args:
      partitions: The `PartitionInfo`s that specifies the mapping between the
        single tensor to the dictionary of tensors.
      action_spec: The ActionSpec that will be exposed by the environment. The
        input (i.e. the action) to the conversion function will need to match
        this spec.
    """
    self._partitions = partitions
    self._action_spec = action_spec

    self._check_partitions_range()

  def _check_partitions_range(self) -> None:
    # We currently consider only 1D tensors.
    if self._action_spec.shape:
      length = self._action_spec.shape[0]
    else:
      length = 0
    for partition in self._partitions:
      if partition.start_index >= length:
        raise ValueError(
            'Invalid partition start index:'
            f' {partition.start_index}({partition.length}). Maximum length:'
            f' {length}.'
        )

      if partition.length is not None:
        if partition.start_index + partition.length > length:
          raise ValueError(
              'Invalid partition:'
              f' {partition.start_index}({partition.length}) ->'
              f' {partition.start_index + partition.length}. Maximum length:'
              f' {length}.'
          )

  def commands_from_environment_action(
      self, environment_action: gdmr_types.ActionType
  ) -> Mapping[str, gdmr_types.ArrayType]:
    """Converts the environment action into commands accepts by REAF."""
    commands = {}
    for info in self._partitions:
      if info.length is None:
        end = None
      else:
        end = info.start_index + info.length
      commands[info.command_key] = np.asarray(
          environment_action[info.start_index : end]
      )
    return commands

  def action_spec(self) -> gdmr_types.ActionSpec:
    """Returns the action spec exposed by the environment."""
    return self._action_spec

  def task_commands_keys(self) -> set[str]:
    """Returns the commands keys as converted by this adapter."""
    return {info.command_key for info in self._partitions}
