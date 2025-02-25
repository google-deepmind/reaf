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
"""Commands processor to clip the commands sent by the agent."""

from collections.abc import Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from typing_extensions import override


class ClipCommandsProcessor(commands_processor.CommandsProcessor):
  """Commands processor to clip the commands sent by the agent.

  This processor clips a command to a static range.

  Examples:
    - To limit the command 'force' to the range [0.5, 1.5], we can use:
      ```
      ClipCommandsProcessor(
          name='force_clipper',
          command_to_clip='force',
          min_command=np.array([0.5]),
          max_command=np.array([1.5]),
      )
      ```

    - To clip a command 'velocity' to a fixed range [0, 1]:
      ```
      ClipCommandsProcessor(
          name='velocity_clipper',
          command_to_clip='velocity',
          min_command=np.array([0.0, 0.0, 0.0]),
          max_command=np.array([1.0, 1.0, 1.0]),
      )
      ```
      The command will always be clipped to the fixed range [0.0, 1.0] for all
      dimensions.
  """

  def __init__(
      self,
      name: str,
      command_to_clip: str,
      min_command: np.ndarray,
      max_command: np.ndarray,
  ):
    """Initializes the clip commands processor.

    Args:
      name: The unique name of this object.
      command_to_clip: The key of the command to be clipped.
      min_command: The minimum allowed value for the command.
      max_command: The maximum allowed value for the command.

    Raises:
      ValueError: If the shapes of the min and max commands do not match, or if
        the shape of the reference value does not match the shape of the
        expected commands.
    """
    self._name = name
    self._command_to_clip = command_to_clip
    self._min_command = min_command
    self._max_command = max_command

    expected_shape = min_command.shape
    if max_command.shape != expected_shape:
      raise ValueError(
          "The shape of the min and max commands do not match. "
          f"Got {max_command.shape=} and {min_command.shape=}."
      )

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    clipped_command = consumed_commands[self._command_to_clip]

    return {
        self._command_to_clip: np.clip(
            clipped_command,
            self._min_command,
            self._max_command,
        )
    }

  @override
  def consumed_commands_spec(self) -> Mapping[str, specs.BoundedArray]:
    """Spec of the commands consumed by this processor."""
    return {
        self._command_to_clip: specs.BoundedArray(
            shape=self._min_command.shape,
            dtype=self._min_command.dtype,
            minimum=self._min_command,
            maximum=self._max_command,
            name=self._command_to_clip,
        )
    }

  @override
  def produced_commands_keys(self) -> set[str]:
    """Keys of the commands produced by this processor."""
    return set([self._command_to_clip])


