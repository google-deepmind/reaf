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
"""Test the RotationMatrixToQuaternion feature."""

from collections.abc import Mapping

from dm_env import specs
import numpy as np
from reaf.common import rotation_matrix_to_quaternion_feature_producer

from absl.testing import absltest

_TEST_NAME = "rotation_matrix_to_quaterion_features_producer"
_TEST_ROTATION_MATRIX_KEY = "rotation_matrix_key"
_TEST_QUATERNION_KEY = "quaternion_key"
_TEST_QUATERNION_SPEC = specs.BoundedArray(
    shape=(4,), dtype=np.float32, minimum=-1.0, maximum=1.0
)
_TEST_ROTATION_MATRIX = [
    [0.2762886, -0.6508952, 0.7071068],
    [0.9353249, 0.0129355, -0.3535534],
    [0.2209795, 0.7590573, 0.6123725],
]
_TEST_QUATERNION = [0.6894919, 0.4034169, 0.1762629, 0.5751409]
_TEST_REQUIRED_FEATURES = required_features = {
    _TEST_ROTATION_MATRIX_KEY: np.asarray(_TEST_ROTATION_MATRIX)
}


def get_feature_producer() -> (
    rotation_matrix_to_quaternion_feature_producer.RotationMatrixToQuaterionFeaturesProducer
):
  return rotation_matrix_to_quaternion_feature_producer.RotationMatrixToQuaterionFeaturesProducer(
      name=_TEST_NAME,
      rotation_matrix_key=_TEST_ROTATION_MATRIX_KEY,
      quaternion_key=_TEST_QUATERNION_KEY,
  )


class RotationMatrixToQuaternionFeatureProducerTest(absltest.TestCase):

  def test_name(self):
    feature = get_feature_producer()
    self.assertEqual(feature.name, _TEST_NAME)

  def test_required_features_keys(self):
    feature = get_feature_producer()
    self.assertSetEqual(
        feature.required_features_keys(), set(_TEST_REQUIRED_FEATURES.keys())
    )

  def test_produced_features_spec(self):
    feature = get_feature_producer()
    self.assertEqual(
        feature.produced_features_spec(),
        {_TEST_QUATERNION_KEY: _TEST_QUATERNION_SPEC},
    )

  def test_produce_features(self):
    feature = get_feature_producer()
    features: Mapping[str, np.ndarray] = feature.produce_features(
        _TEST_REQUIRED_FEATURES
    )
    expected_features: Mapping[str, np.ndarray] = {
        _TEST_QUATERNION_KEY: np.array(_TEST_QUATERNION)
    }
    self.assertLen(features, len(expected_features))
    self.assertIn(_TEST_QUATERNION_KEY, features)
    np.testing.assert_array_almost_equal(
        features[_TEST_QUATERNION_KEY], expected_features[_TEST_QUATERNION_KEY]
    )


if __name__ == "__main__":
  absltest.main()
