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
"""Features producer that concatenates features along a specified axis."""

from collections.abc import Mapping
import functools

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import features_producer


class ConcatenateFeaturesProducer(features_producer.FeaturesProducer):
  """Concatenates a list of features along the desired axis.

  We expect features to have the same number of dimensions.

  Examples:
    - Given the 2 features `{'feature1':[1, 2, 3]` and `'feature2':[4, 5, 6]}`,
      we can create a new feature `{'feature3': [1, 2, 3, 4, 5, 6]}` using
      ConcatenateFeaturesProducer(
          name='foo',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(3,), dtype=np.int32),
              'feature2': specs.Array(shape=(3,), dtype=np.int32)
          },
          new_feature_name='feature3',
      )
    - Given the 2 features `{'feature1':[[1, 2, 3], [4, 5, 6]]` and
      `'feature2':[[7, 8, 9], [10, 11, 12]]}`,
      we can create a new feature
      `{'feature3': [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]}` using
      ConcatenateFeaturesProducer(
          name='foo',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(2, 3), dtype=np.int32),
              'feature2': specs.Array(shape=(2, 3), dtype=np.int32)
          },
          new_feature_name='feature3',
          axis=0
      )
    - Given the 2 features `{'feature1':[[1, 2, 3], [4, 5, 6]]` and
      `'feature2':[[7, 8, 9], [10, 11, 12]]}`,
      we can create a new feature
      `{'feature3': [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]}` using
      ConcatenateFeaturesProducer(
          name='foo',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(2, 3), dtype=np.int32),
              'feature2': specs.Array(shape=(2, 3), dtype=np.int32)
          },
          new_feature_name='feature3',
          axis=1
      )
    - Given the 2 features `{'feature1':[[1, 2, 3], [4, 5, 6]]` and
      `'feature2':[[7, 8, 9], [10, 11, 12]]}`,
      we can create a new feature
      `{'feature3': [1, 2, 3, 4, 5, 6, 7, 8 ,9, 10, 11, 12]}` using
      ConcatenateFeaturesProducer(
          name='foo',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(2, 3), dtype=np.int32),
              'feature2': specs.Array(shape=(2, 3), dtype=np.int32)
          },
          new_feature_name='feature3',
          axis=None
      )

  This class will raise an error if:
  - The features to concatenate have different sizes except for the axis along
    which the concatenation is performed except if `axis is None`.
  - The features to concatenate have different dtypes.
  - The features to concatenate have different number of dimensions.
  """

  def __init__(
      self,
      name: str,
      concatenated_features_specs: Mapping[str, specs.Array],
      new_feature_name: str,
      axis: int | None = 0,
  ):
    """Constructor.

    Args:
      name: Name of the features producer.
      concatenated_features_specs: Mapping for the feature key to the spec of
        that feature. We expected all concatenated features to have the same
        dtype.
      new_feature_name: Name of the new feature.
      axis: Similar to the axis parameter of np.concatenate. The axis along
        which the arrays will be joined. If axis is None, arrays are flattened
        before use. Default is 0.

    Raises:
      ValueError: If the features to concatenate have different size dimensions
        except for the axis along which the concatenation is performed.
      ValueError: If the features to concatenate have different number of
        dimensions.
      TypeError: If the features to concatenate have different dtypes.
    """
    self._name = name
    self._concatenated_features_specs = concatenated_features_specs
    self._new_feature_name = new_feature_name
    self._axis = axis
    self._dtype = next(iter(concatenated_features_specs.values())).dtype
    self._shape = None
    self._check_dtypes_and_compute_shape()

  @property
  def name(self) -> str:
    """Returns a unique string identifier for this object."""
    return self._name

  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    return {
        self._new_feature_name: np.concatenate(
            [
                required_features[feature_name]
                for (feature_name) in self._concatenated_features_specs
            ],
            axis=self._axis,
            dtype=self._dtype,
        )
    }

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    """Returns the spec of the features produced by this producer."""
    return {
        self._new_feature_name: specs.Array(
            shape=self._shape, dtype=self._dtype
        )
    }

  def required_features_keys(self) -> set[str]:
    """Returns the keys that are required to produce the new features."""
    return set(self._concatenated_features_specs.keys())

  def _check_dtypes_and_compute_shape(self):
    """Checks that all features to concatenate have the same dtype and shape."""
    if self._axis is None:
      new_shape = 0
      for feature_spec in self._concatenated_features_specs.values():
        new_shape += functools.reduce(lambda x, y: x * y, feature_spec.shape)
      # The shape needs to be a tuple.
      self._shape = (new_shape,)
    else:
      new_shape = None
      for (
          feature_name,
          feature_spec,
      ) in self._concatenated_features_specs.items():
        if new_shape is None:
          new_shape = list(feature_spec.shape)
        # To compute the new shape we add the size of the feature along the
        # concatenation axis. All the other dimensions are kept the same as the
        # ones in the features. For example if we have 3 features with shape
        # (2, 3, 4), (4, 3, 4), (1, 3, 4) and we concatenate along the 0th
        # dimension, the new shape will be (7, 3, 4).
        else:
          new_shape[self._axis] += feature_spec.shape[self._axis]
          for i, shape_i in enumerate(feature_spec.shape):
            if len(new_shape) != len(feature_spec.shape):
              raise ValueError(
                  'All features to concatenate must have the same number of'
                  f' dimensions. Was expecting {len(new_shape)} dimensions but'
                  f' got {feature_spec.shape} for feature {feature_name}.'
              )
            if i != self._axis and shape_i != new_shape[i]:
              raise ValueError(
                  'All features to concatenate must have the same shape except'
                  f' the {self._axis}th dimension. Was expecting'
                  f' {new_shape} but got {feature_spec.shape} for feature'
                  f' {feature_name}.'
              )
        self._shape = tuple(new_shape)

    new_dtype = None
    for feature_name, feature_spec in self._concatenated_features_specs.items():
      if new_dtype is None:
        new_dtype = feature_spec.dtype
      else:
        if feature_spec.dtype != new_dtype:
          raise TypeError(
              'All features to concatenate must have the same dtype. Was'
              f' expecting {new_dtype} but got'
              f' {feature_spec.dtype} for feature'
              f' {feature_name}.'
          )
    self._dtype = new_dtype
