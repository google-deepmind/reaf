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
"""Features producer that stacks a set of features along the time axis."""

from collections.abc import Mapping, Sequence
import dataclasses

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from numpy import typing
from reaf.core import features_producer


@dataclasses.dataclass(frozen=True, kw_only=True)
class StackedFeatureConfig:
  """Configuration for stacked feature.

  Attributes:
    feature_name: Name of the feature to be stacked.
    stacked_feature_name: Name of the stacked feature.
    feature_shape: Shape of the feature to be stacked.
    dtype: Type of the elements in the array.
  """

  feature_name: str
  stacked_feature_name: str
  feature_shape: tuple[int, ...]
  dtype: typing.DTypeLike


class StackFeaturesProducer(features_producer.FeaturesProducer):
  """Stacks a set of features for a given number of steps along the time axis.

  This features producer allows to stack successive features for a given number
  of steps. For the first step, the feature is copied `stack_size` times. For
  the following steps, the feature is prepended to the list and the oldest
  feature is removed.

  This features producer is useful to create a history of features that can be
  used by the model to infer how the system is behaving in time.

  Example:
    Given a feature `feature_1` of shape (3, 2). If the values of this feature
      are:
      - Step 1: [[1, 2], [3, 4], [5, 6]]
      - Step 2: [[7, 8], [9, 10], [11, 12]]
      - Step 3: [[13, 14], [15, 16], [17, 18]]
    If we use a stack size of 2, the stacked feature will have a shape of
    (2, 3, 2) and be:
      - Step 1: [[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]]
      - Step 2: [[[7, 8], [9, 10], [11, 12]], [[1, 2], [3, 4], [5, 6]]]
      - Step 3: [[[13, 14], [15, 16], [17, 18]], [[7, 8], [9, 10], [11, 12]]]
  """

  def __init__(
      self,
      name: str,
      stacked_features_configs: Sequence[StackedFeatureConfig],
      stack_size: int,
  ):
    """Constructor.

    Args:
      name: Name of the features producer.
      stacked_features_configs: Sequence of stacked features configuration.
      stack_size: Number of times the features should be stacked. Should be
        greater than 1.
    """
    self._name = name
    self._stacked_features_configs = stacked_features_configs
    self._stacked_features: dict[str, np.ndarray] = {}
    self._stack_size = stack_size
    self._first_step_setup_needed = True
    if stack_size < 2:
      raise ValueError("Stack size should be greater than 1.")
    self.reset()

  @property
  def name(self) -> str:
    """Returns a unique string identifier for this object."""
    return self._name

  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    """Produces the stacked features."""
    if self._first_step_setup_needed:
      for config in self._stacked_features_configs:
        self._stacked_features[config.stacked_feature_name] = np.tile(
            required_features[config.feature_name],
            # We want to add a new axis to the tiled data. We must tile
            # the data along the a new axis and along None of the previous ones.
            # We need to have a tiling of (stack_size, 1, 1, ..., 1).
            (
                self._stack_size,
                *([1] * len(required_features[config.feature_name].shape)),
            ),
        )
      self._first_step_setup_needed = False
    else:
      for config in self._stacked_features_configs:
        self._stacked_features[config.stacked_feature_name] = np.roll(
            self._stacked_features[config.stacked_feature_name],
            1,
            axis=0,
        )
        self._stacked_features[config.stacked_feature_name][0] = (
            required_features[config.feature_name]
        )
    return self._stacked_features

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    """Returns the spec of the features produced by this producer."""
    features_specs = {}
    for config in self._stacked_features_configs:
      features_specs[config.stacked_feature_name] = specs.Array(
          shape=(self._stack_size, *config.feature_shape),
          dtype=config.dtype,
      )
    return features_specs

  def required_features_keys(self) -> set[str]:
    """Returns the keys that are required to produce the new features."""
    return set(config.feature_name for config in self._stacked_features_configs)

  def reset(self):
    """Resets the stacked features to empty numpy arrays."""
    self._first_step_setup_needed = True
    for config in self._stacked_features_configs:
      self._stacked_features[config.stacked_feature_name] = np.empty(
          (self._stack_size, *config.feature_shape)
      )
