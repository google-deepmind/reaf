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
"""Features producer that flattens a set of features."""

from collections.abc import Mapping, Sequence
import dataclasses
import strenum

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy.typing as npt
from reaf.core import features_producer


class Order(strenum.StrEnum):
  """Order of numpy arrays."""

  # The numpy flatten method takes an order parameter. 'C' means to flatten in
  # row-major (C-style) order. 'F' means to latten in column-major
  # (Fortran-style) order.
  COLUMN_MAJOR = "F"
  ROW_MAJOR = "C"


@dataclasses.dataclass(frozen=True, kw_only=True)
class FlattenedFeatureConfig:
  """Configuration feature.

  Attributes:
    feature_name: Name of the feature to be flattened.
    flattened_feature_name: Name of the flattened feature.
    flattened_size: Number of elements in the feature.
    dtype: Type of the elements in the array.
    order: If the array is row major or column major.
  """

  feature_name: str
  flattened_feature_name: str
  flattened_size: int
  dtype: npt.DTypeLike
  order: Order = Order.ROW_MAJOR


class FlattenFeaturesProducer(features_producer.FeaturesProducer):
  """Flattens set of features.

  Examples:
    - Given the feature `{'feature1':[[1, 2, 3], [4, 5, 6]]}`,
      we can create a new feature `{'feature1_flattened': [1, 2, 3, 4, 5, 6]}`
      using
      FlattenFeaturesProducer(
          name='foo',
          flattened_features_configs=[
              'feature1': FlattenedFeatureConfig(
                  feature_name='feature1',
                  flattened_feature_name='feature1_flattened',
                  flattened_size=6,
                  dtype=np.int32
              )
          ]
      )
  """

  def __init__(
      self,
      name: str,
      flattened_features_configs: Sequence[FlattenedFeatureConfig],
  ):
    """Constructor.

    Args:
      name: Name of the features producer.
      flattened_features_configs: Sequence of flattened features configuration.
    """
    self._name = name
    self._flattened_features_configs = flattened_features_configs

  @property
  def name(self) -> str:
    """Returns a unique string identifier for this object."""
    return self._name

  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    new_features = {}
    for config in self._flattened_features_configs:
      new_features[config.flattened_feature_name] = required_features[
          config.feature_name
      ].flatten(order=config.order)
    return new_features

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    """Returns the spec of the features produced by this producer."""
    features_specs = {}
    for config in self._flattened_features_configs:
      features_specs[config.flattened_feature_name] = specs.Array(
          shape=(config.flattened_size,), dtype=config.dtype
      )
    return features_specs

  def required_features_keys(self) -> set[str]:
    """Returns the keys that are required to produce the new features."""
    return set(
        config.feature_name for config in self._flattened_features_configs
    )
