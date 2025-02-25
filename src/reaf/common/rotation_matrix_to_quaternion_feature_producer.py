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
"""Feature Producer for computing the quaternion from the rotation matrix."""

from collections.abc import Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import features_producer
import scipy.spatial.transform

_EXPECTED_SHAPE = (3, 3)


class RotationMatrixToQuaterionFeaturesProducer(
    features_producer.FeaturesProducer
):
  """Converts a rotation matrix to a quaternion (WXYZ).

  Example:
    RotationMatrixToQuaterionFeaturesProducer(
        name='rotation_matrix_to_quaterion_features_producer',
        rotation_matrix_key='panda_tcp_site_rotation_matrix',
        quaternion_key='panda_tcp_quaternion',
    )
  """

  def __init__(
      self,
      name: str,
      rotation_matrix_key: str,
      quaternion_key: str,
  ):
    self._name = name
    self._rotation_matrix_key = rotation_matrix_key
    self._quaternion_key = quaternion_key

  @property
  def name(self) -> str:
    return self._name

  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    """Returns the quaternion of a given rotation matrix."""
    if (
        np.asarray(required_features[self._rotation_matrix_key]).shape
        != _EXPECTED_SHAPE
    ):
      raise ValueError(
          f'The rotation matrix should be of shape {_EXPECTED_SHAPE}, but got'
          f' {np.asarray(required_features[self._rotation_matrix_key]).shape}.'
      )
    rotation = scipy.spatial.transform.Rotation.from_matrix(
        required_features[self._rotation_matrix_key]
    )
    return {self._quaternion_key: rotation.as_quat(scalar_first=True)}

  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    """Returns the spec of the features produced by this producer."""
    return {
        self._quaternion_key: specs.BoundedArray(
            shape=(4,), dtype=np.float32, minimum=-1, maximum=1
        )
    }

  def required_features_keys(self) -> set[str]:
    """Returns the keys that are required to produce the new features."""
    return set([self._rotation_matrix_key])
