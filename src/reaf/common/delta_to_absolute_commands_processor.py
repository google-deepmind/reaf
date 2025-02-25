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
"""Commands processor used to expose a delta command action space."""

from collections.abc import Mapping
import dataclasses

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import commands_processor
from reaf.core import features_observer
from typing_extensions import override


@dataclasses.dataclass
class DeltaToAbsoluteConfig:
  """Configuration for a delta to absolute commands processor.

  Attributes:
    consumed_command_key: The key of the consumed command.
    consumed_command_spec: The spec of the consumed command.
    produced_command_key: The key of the produced command.
    reference_feature_key: Key of the feature that is used as reference to which
      the delta command is added.
  """

  consumed_command_key: str
  consumed_command_spec: specs.BoundedArray
  produced_command_key: str
  reference_feature_key: str


class DeltaToAbsoluteCommandsProcessor(
    commands_processor.CommandsProcessor, features_observer.FeaturesObserver
):
  """Command processor for delta commands based on an observered feature.

  This commands processor is used to expose a delta command action space.
  At each step, the delta command is added to an observed reference feature.
  """

  def __init__(
      self,
      name: str,
      config: DeltaToAbsoluteConfig,
      *,
      validate_reference_feature: bool = True,
  ):
    super().__init__()
    self._name = name
    self._config = config
    self._reference = np.zeros(self._config.consumed_command_spec.shape)

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
    return {
        self._config.produced_command_key: (
            self._reference
            + consumed_commands[self._config.consumed_command_key]
        )
    }

  @override
  def consumed_commands_spec(self) -> dict[str, specs.BoundedArray]:
    return {
        self._config.consumed_command_key: self._config.consumed_command_spec
    }

  @override
  def produced_commands_keys(self) -> set[str]:
    return set([self._config.produced_command_key])

  def _maybe_validate_reference_feature(
      self, reference_feature: gdmr_types.ArrayType
  ):
    if not self._validated_reference_feature:
      if reference_feature.shape != self._config.consumed_command_spec.shape:
        raise ValueError(
            f'Shape of reference feature {self._config.reference_feature_key} '
            f'({reference_feature.shape}) does not match the shape of the '
            f'consumed command ({self._config.consumed_command_spec.shape}).'
        )
      if reference_feature.dtype != self._config.consumed_command_spec.dtype:
        raise ValueError(
            f'Dtype of reference feature {self._config.reference_feature_key} '
            f'({reference_feature.dtype}) does not match the dtype of the '
            f'consumed command ({self._config.consumed_command_spec.dtype}).'
        )
    self._validated_reference_feature = True

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
        features[self._config.reference_feature_key]
    )
    self._reference = features[self._config.reference_feature_key]
