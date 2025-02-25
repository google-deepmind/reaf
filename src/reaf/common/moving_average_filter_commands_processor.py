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
"""Commands processor to filter commands using a moving average."""

from collections.abc import Mapping, Sequence
import dataclasses

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from typing_extensions import override


@dataclasses.dataclass
class MovingAverageFilterConfig:
  """Configuration for a filter commands processor.

  Attributes:
    commands_to_filter:  Keys of commands that will be filtered.
    commands_spec: Specs of the commands to be filtered.
    average_window_size:  Size of the averaging window.
  """

  commands_to_filter: Sequence[str]
  commands_spec: Mapping[str, specs.BoundedArray]
  average_window_size: int


class MovingAverageFilterCommandsProcessor(
    commands_processor.CommandsProcessor
):
  """Command processor for filtering commands using a moving average."""

  def __init__(
      self,
      name: str,
      config: MovingAverageFilterConfig,
  ):
    super().__init__()
    self._name = name
    self._config = config
    self._commands_history: dict[str, list[np.ndarray]] = {
        key: [] for key in self._config.commands_to_filter
    }
    self._validated_specs = False

    # Validate that the provided specs match the commands_to_filter.
    if set(self._config.commands_to_filter) != set(
        self._config.commands_spec.keys()
    ):
      raise ValueError(
          "The keys in 'commands_to_filter' and 'commands_spec' must match."
          f" Got commands_to_filter={self._config.commands_to_filter} and"
          f" commands_spec keys={list(self._config.commands_spec.keys())}."
      )

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    output_commands = {}

    for key, value in consumed_commands.items():
      if not self._validated_specs:
        # Validate the shape and dtype of incoming commands (only once)
        expected_spec = self._config.commands_spec[key]
        if value.shape != expected_spec.shape:
          raise ValueError(
              f"Shape mismatch for command '{key}'. Expected"
              f" {expected_spec.shape}, but got {value.shape}."
          )
        if value.dtype != expected_spec.dtype:
          raise ValueError(
              f"Dtype mismatch for command '{key}'. Expected"
              f" {expected_spec.dtype}, but got {value.dtype}."
          )
      self._commands_history[key].append(value)

      if len(self._commands_history[key]) > self._config.average_window_size:
        self._commands_history[key].pop(0)

      output_commands[key] = np.mean(
          self._commands_history[key], axis=0
      )  # Filtered value

    self._validated_specs = True

    return output_commands

  @override
  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return self._config.commands_spec

  @override
  def produced_commands_keys(self) -> set[str]:
    return set(self._config.commands_to_filter)

  @override
  def reset(self) -> None:
    """Resets the internal state of the command processor."""
    for key in self._commands_history:
      self._commands_history[key] = []
    self._validated_specs = False  # Reset the flag
