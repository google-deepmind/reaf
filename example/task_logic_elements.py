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
"""Elements (processors, producers, etc) for the task logic layer."""

from collections.abc import Callable, Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from reaf.core import features_producer
from reaf.core import reward_provider
import tree
from typing_extensions import override


POSITION_REFERENCE_KEY = "current_reference"


class ConstantFactorCommandsProcessor(commands_processor.CommandsProcessor):
  """Substitute a command with its copy multiplied by a constant factor."""

  def __init__(
      self,
      consumed_spec: gdmr_types.AnyArraySpec,
      consumed_key: str,
      produced_key: str,
      factor: float,
  ):
    """Initializer.

    Args:
      consumed_spec: Command spec consumed by this processor.
      consumed_key: Key associated with the consumed spec.
      produced_key: Key associated with the produced command,
      factor: multiplicative factor used to produced the new command.
    """
    self._consumed_spec = consumed_spec
    self._consumed_key = consumed_key
    self._produced_key = produced_key
    self._torque_to_current_factor = factor

  @override
  @property
  def name(self) -> str:
    return "TorquesToCurrentCommandProcessor"

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    command = consumed_commands[self._consumed_key]
    return {self._produced_key: command * self._torque_to_current_factor}

  @override
  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return {self._consumed_key: self._consumed_spec}

  @override
  def produced_commands_keys(self) -> set[str]:
    return set([self._produced_key])


class ReferenceProducer(features_producer.FeaturesProducer):
  """Produces a feature corresponding to the current position reference."""

  def __init__(self, reference_producer: Callable[[], np.ndarray]):
    """Initializer.

    Args:
      reference_producer: Callable that returns the current position reference.
        This will be called every time `compute_all_features` is called on the
        TaskLogicLayer, i.e. at every environment step.
    """
    self._producer = reference_producer

  @property
  def name(self) -> str:
    return "_ReferenceProducer"

  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    new_reference = self._producer()
    return {POSITION_REFERENCE_KEY: new_reference}

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    # Get a reference to retrieve its size.
    reference = self._producer()
    return {
        POSITION_REFERENCE_KEY: specs.Array(
            shape=reference.shape, dtype=reference.dtype
        )
    }

  def required_features_keys(self) -> set[str]:
    return set()


class PositionAndVelocityErrorRewardProvider(reward_provider.RewardProvider):
  """Computes a reward based on the position error and current velocity.

  Position and velocity terms are weighted differently.
  Reward = 1 - (reference - position)^2 - 0.1 velocity^2.
  """

  def __init__(
      self, position_reference_key: str, position_key: str, velocity_key: str
  ):
    self._position_reference_key = position_reference_key
    self._position_key = position_key
    self._velocity_key = velocity_key

  @override
  def name(self) -> str:
    return "_PositionAndVelocityErrorRewardProvider"

  @override
  def compute_reward(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> tree.Structure[gdmr_types.ArrayType]:
    position_error = (
        required_features[self._position_reference_key]
        - required_features[self._position_key]
    )
    position_square_norm = np.linalg.norm(position_error) ** 2
    velocity_square_norm = (
        np.linalg.norm(required_features[self._velocity_key]) ** 2
    )

    return 1 - (position_square_norm + 0.1 * velocity_square_norm)

  @override
  def reward_spec(self) -> tree.Structure[specs.Array]:
    return specs.Array(shape=(1,), dtype=float)

  @override
  def required_features_keys(self) -> set[str]:
    """Returns the feature keys that are required to compute the reward."""
    return set(
        [self._position_reference_key, self._position_key, self._velocity_key]
    )
