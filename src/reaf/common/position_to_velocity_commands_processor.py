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
"""Commands processor to transform position commands into velocity commands."""

from collections.abc import Mapping
import dataclasses

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from reaf.core import features_observer
from typing_extensions import override


def _compute_velocity_command(
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


@dataclasses.dataclass
class PositionToVelocityConfig:
  """Configuration for a position to velocity commands processor.

  Attributes:
    consumed_position_command_key: The key of the consumed position command.
    consumed_position_command_spec: The spec of the consumed position command.
    produced_velocity_command_key: The key of the produced velocity command.
    position_reference_feature_key: Key of the feature that is used as the
      current position.
    minimum_velocity_command: The minimum allowed velocity command.
    maximum_velocity_command: The maximum allowed velocity command.
    feedback_gains: The feedback gains of the P-controller. The desired velocity
      is computed as the feedback gains times the position error.
    max_acceleration: The maximum allowed acceleration. Meaning the maximum
      change in velocity per second.
    control_timestep: The time step between two calls of the `process_commands`
      method
  """

  consumed_position_command_key: str
  consumed_position_command_spec: specs.BoundedArray
  produced_velocity_command_key: str
  position_reference_feature_key: str
  minimum_velocity_command: np.ndarray
  maximum_velocity_command: np.ndarray
  feedback_gains: np.ndarray
  max_acceleration: np.ndarray
  control_timestep: float


class PositionToVelocityCommandsProcessor(
    commands_processor.CommandsProcessor, features_observer.FeaturesObserver
):
  """Command processor to transform position commands into velocity commands.

  This commands processor takes in a position command and transforms it into a
  velocity command using a P-controller.
  """

  def __init__(
      self,
      name: str,
      config: PositionToVelocityConfig,
      *,
      validate_reference_feature: bool = True,
  ):
    super().__init__()
    self._name = name
    self._config = config
    self._current_position = np.zeros(
        self._config.consumed_position_command_spec.shape
    )
    self._current_velocity_command = np.zeros(
        self._config.consumed_position_command_spec.shape
    )

    # We reuse _validated_reference_feature as a variable to decide if we should
    # validate the reference feature or not. Note that the meaning is different:
    # `_validated_reference_feature` means that we already validated the specs.
    # So we negate the configuration option: If
    # `validate_reference_feature == True`, we set it to False, meaning that we
    # will still need to validate the specs. If
    # `validate_reference_feature == False` we set it to True, meaning that we
    # will have already validated the specs, which will not trigger the actual
    # validation.
    self._validated_reference_feature = not validate_reference_feature

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    self._current_velocity_command = _compute_velocity_command(
        desired_positions=consumed_commands[
            self._config.consumed_position_command_key
        ],
        current_positions=self._current_position,
        current_velocity_command=self._current_velocity_command,
        minimum_velocity_command=self._config.minimum_velocity_command,
        maximum_velocity_command=self._config.maximum_velocity_command,
        feedback_gains=self._config.feedback_gains,
        max_acceleration=self._config.max_acceleration,
        control_timestep=self._config.control_timestep,
    )
    return {
        self._config.produced_velocity_command_key: (
            self._current_velocity_command
        )
    }

  @override
  def consumed_commands_spec(self) -> dict[str, specs.BoundedArray]:
    return {
        self._config.consumed_position_command_key: (
            self._config.consumed_position_command_spec
        )
    }

  @override
  def produced_commands_keys(self) -> set[str]:
    return set([self._config.produced_velocity_command_key])

  def _maybe_validate_reference_feature(
      self, reference_feature: gdmr_types.ArrayType
  ):
    if not self._validated_reference_feature:
      if (
          reference_feature.shape
          != self._config.consumed_position_command_spec.shape
      ):
        raise ValueError(
            'Shape of reference feature'
            f' {self._config.position_reference_feature_key} ({reference_feature.shape})'
            ' does not match the shape of the consumed command'
            f' ({self._config.consumed_position_command_spec.shape}).'
        )
      if (
          reference_feature.dtype
          != self._config.consumed_position_command_spec.dtype
      ):
        raise ValueError(
            'Dtype of reference feature'
            f' {self._config.position_reference_feature_key} ({reference_feature.dtype})'
            ' does not match the dtype of the consumed command'
            f' ({self._config.consumed_position_command_spec.dtype}).'
        )
    self._validated_reference_feature = True

  @override
  def observe_features(self, features: Mapping[str, gdmr_types.ArrayType]):
    """Observes the features and updates the reference value.

    Args:
      features: The features to observe. The observed feature must have the same
        shape and dtype as the consumed command, we expect the feature to be
        the current position.

    Raises:
      ValueError: If the shape of the reference feature is different to the
        consummed command shape.
      ValueError: If the dtype of the reference feature is different to the
        consummed command dtype.
    """
    self._maybe_validate_reference_feature(
        features[self._config.position_reference_feature_key]
    )
    self._current_position = features[
        self._config.position_reference_feature_key
    ]
