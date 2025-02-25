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

from absl import logging
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from reaf.core import features_observer
from typing_extensions import override


class RelativeClipCommandsProcessor(
    commands_processor.CommandsProcessor, features_observer.FeaturesObserver
):
  """Commands processor to clip the relative change in commands sent by the agent.

  This processor clips a command to a range relative to a specified value
  obtained from a reference feature.

  Attributes:
    name: The unique name of this object.
    command_to_clip: The key of the consumed command.
    command_spec: The spec of the consumed command.
    delta_lower: The lower bound of the relative change.
    delta_upper: The upper bound of the relative change.
    reference_feature_key: Key of the feature that is used as reference to which
      the delta command is added.
    validate_reference_feature: Whether to validate the reference feature.

  Examples:
    - To limit the command 'position' to the maximum change of 0.2 from the
      previous position command, we can use:
      ```
      RelativeClipCommandsProcessor(
          name='position_clipper',
          command_to_clip='position',
          command_spec=specs.BoundedArray(
              shape=(1,),
              dtype=np.float32,
              minimum=np.array([0.0]),
              maximum=np.array([2.0]),
          ),
          delta_lower=np.array([0.2]),
          delta_upper=np.array([0.2]),
          reference_feature_key='observed_position',
      )
      ```
      where `observed_position` is a feature key passed to the observer.
      This means that if the associated value to `observed_position` is 1.0,
      the processor will clip the command to be in the range [0.8, 1.2].
  """

  def __init__(
      self,
      name: str,
      command_to_clip: str,
      command_spec: specs.BoundedArray,
      delta_lower: np.ndarray,
      delta_upper: np.ndarray,
      reference_feature_key: str,
  ):
    super().__init__()
    if delta_lower.shape != delta_upper.shape:
      raise ValueError(
          'Shape of delta_lower and delta_upper must match.'
      )
    if not np.all(delta_lower > 0) or not np.all(delta_upper > 0):
      raise ValueError('Delta must be positive.')

    self._name = name
    self._command_to_clip = command_to_clip
    self._command_spec = command_spec
    self._delta_lower = delta_lower
    self._delta_upper = delta_upper
    self._reference_feature_key = reference_feature_key

    self._reference = np.zeros(self._command_spec.shape)

    self._validated_reference_feature = False
    self._validated_command = False

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:

    command = consumed_commands[self._command_to_clip]
    self._maybe_validate_command(command)
    clipped_command = np.clip(
        command,
        self._reference - self._delta_lower,
        self._reference + self._delta_upper,
    )
    if not np.allclose(clipped_command, command):
      logging.warning('clipping command (relative): %s', self._command_to_clip)

    return {self._command_to_clip: clipped_command}

  @override
  def consumed_commands_spec(self) -> dict[str, specs.BoundedArray]:
    return {self._command_to_clip: self._command_spec}

  @override
  def produced_commands_keys(self) -> set[str]:
    return set([self._command_to_clip])

  def _maybe_validate_command(
      self, consumed_command: gdmr_types.ArrayType
  ):
    if not self._validated_command:
      if consumed_command.shape != self._command_spec.shape:
        raise ValueError(
            f'Shape of the command {self._command_to_clip} '
            f'({consumed_command.shape}) does not match the shape of the '
            f'consumed command ({self._command_spec.shape}).'
        )
      if not self._dtypes_loosely_matches(
          consumed_command.dtype, self._command_spec.dtype
      ):
        raise ValueError(
            f'Dtype of the consumed command {self._command_to_clip} '
            f'({consumed_command.dtype}) does not match the dtype of the '
            f'command ({self._command_spec.dtype}).'
        )
    self._validated_command = True

  def _maybe_validate_reference_feature(
      self, reference_feature: gdmr_types.ArrayType
  ):
    if not self._validated_reference_feature:
      if reference_feature.shape != self._command_spec.shape:
        raise ValueError(
            f'Shape of reference feature {self._reference_feature_key} '
            f'({reference_feature.shape}) does not match the shape of the '
            f'consumed command ({self._command_spec.shape}).'
        )
      if not self._dtypes_loosely_matches(
          reference_feature.dtype, self._command_spec.dtype
      ):
        raise ValueError(
            f'Dtype of reference feature {self._reference_feature_key} '
            f'({reference_feature.dtype}) does not match the dtype of the '
            f'consumed command ({self._command_spec.dtype}).'
        )
    self._validated_reference_feature = True

  def _dtypes_loosely_matches(self, dtype1: np.dtype, dtype2: np.dtype):
    """Compares two dtypes, allowing for float32 <-> float64 conversions."""
    if dtype1 == dtype2:
      return True
    if dtype1 == np.float32 and dtype2 == np.float64:
      return True
    if dtype1 == np.float64 and dtype2 == np.float32:
      return True
    return False

  @override
  def observe_features(self, features: Mapping[str, gdmr_types.ArrayType]):
    """Observes the features and updates the reference value.

    Args:
      features: The features to observe.

    Raises:
      ValueError: If the shape of the reference feature is different to the
        consummed command shape.
      ValueError: If the dtype of the reference feature is different to the
        consummed command dtype.
    """
    self._maybe_validate_reference_feature(
        features[self._reference_feature_key]
    )
    self._reference = features[self._reference_feature_key]
