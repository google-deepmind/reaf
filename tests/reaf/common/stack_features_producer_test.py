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
from dm_env import specs
import numpy as np
from reaf.common import stack_features_producer
from absl.testing import absltest
from absl.testing import parameterized


class StackFeaturesProducerTest(parameterized.TestCase):

  def test_name(self):
    """Tests name property."""
    producer = stack_features_producer.StackFeaturesProducer(
        name='foo', stacked_features_configs=[], stack_size=4
    )
    self.assertEqual(producer.name, 'foo')

  def test_produce_features_first_step(
      self,
  ):
    """Tests produce_features method."""
    stacked_features_configs = [
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature1',
            stacked_feature_name='feature1_stacked',
            feature_shape=(3,),
            dtype=np.int32,
        ),
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature2',
            stacked_feature_name='feature2_stacked',
            feature_shape=(4, 3),
            dtype=np.float32,
        ),
    ]
    producer = stack_features_producer.StackFeaturesProducer(
        name='test',
        stacked_features_configs=stacked_features_configs,
        stack_size=4,
    )
    required_features = {
        'feature1': np.array([1, 2, 3], dtype=np.int32),
        'feature2': np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32
        ),
    }
    expected_stacked_features = {
        'feature1_stacked': np.tile([1, 2, 3], (4, 1)),
        'feature2_stacked': np.tile(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], (4, 1, 1)
        ),
    }
    stacked_features = producer.produce_features(
        required_features=required_features
    )
    for k, v in stacked_features.items():
      np.testing.assert_array_equal(v, expected_stacked_features[k])

  def test_produce_features_several_steps(
      self,
  ):
    """Tests produce_features method."""
    stacked_features_configs = [
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature1',
            stacked_feature_name='feature1_stacked',
            feature_shape=(3,),
            dtype=np.int32,
        ),
    ]
    producer = stack_features_producer.StackFeaturesProducer(
        name='test',
        stacked_features_configs=stacked_features_configs,
        stack_size=4,
    )
    for i in range(3):
      required_features = {
          'feature1': np.arange(i * 3, (i + 1) * 3),
      }
      producer.produce_features(required_features=required_features)
    stacked_features = producer.produce_features(
        required_features={
            'feature1': np.arange(9, 12),
        }
    )
    np.testing.assert_array_equal(
        stacked_features['feature1_stacked'],
        # Arrange elements from 0 to 11 in a 4x3 matrix. then flip along the
        # first axis to put the features in reverse order.
        np.flip(np.arange(3 * 4).reshape(4, 3), axis=0),
    )

  def test_produced_features_spec(self):
    """Tests produced_features_spec method."""
    stacked_features_configs = [
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature1',
            stacked_feature_name='feature1_stacked',
            feature_shape=(3,),
            dtype=np.int32,
        ),
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature2',
            stacked_feature_name='feature2_stacked',
            feature_shape=(4, 3),
            dtype=np.float32,
        ),
    ]
    producer = stack_features_producer.StackFeaturesProducer(
        name='test',
        stacked_features_configs=stacked_features_configs,
        stack_size=4,
    )

    expected_spec = {
        'feature1_stacked': specs.Array(
            shape=(4, 3), dtype=np.int32, name='feature1_stacked'
        ),
        'feature2_stacked': specs.Array(
            shape=(4, 4, 3), dtype=np.float32, name='feature2_stacked'
        ),
    }
    self.assertEqual(producer.produced_features_spec(), expected_spec)

  def test_produced_features_matches_spec(self):
    """Tests that the produced features adhere to the returned spec."""
    stacked_features_configs = [
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature1',
            stacked_feature_name='feature1_stacked',
            feature_shape=(3,),
            dtype=np.int32,
        ),
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature2',
            stacked_feature_name='feature2_stacked',
            feature_shape=(4, 3),
            dtype=np.float32,
        ),
    ]
    producer = stack_features_producer.StackFeaturesProducer(
        name='test',
        stacked_features_configs=stacked_features_configs,
        stack_size=4,
    )
    produced_features = producer.produce_features(
        required_features={
            'feature1': np.arange(3, dtype=np.int32),
            'feature2': np.arange(4 * 3, dtype=np.float32).reshape(4, 3),
        }
    )
    for key, spec in producer.produced_features_spec().items():
      try:
        spec.validate(produced_features[key])
      except ValueError as e:
        self.fail(e)

  def test_required_features_keys(self):
    """Tests required_features_keys method."""
    stacked_features_configs = [
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature1',
            stacked_feature_name='feature1_stacked',
            feature_shape=(3,),
            dtype=np.int32,
        ),
        stack_features_producer.StackedFeatureConfig(
            feature_name='feature2',
            stacked_feature_name='feature2_stacked',
            feature_shape=(2, 2),
            dtype=np.float64,
        ),
    ]
    producer = stack_features_producer.StackFeaturesProducer(
        name='foo',
        stacked_features_configs=stacked_features_configs,
        stack_size=3,
    )
    self.assertEqual(
        producer.required_features_keys(), {'feature1', 'feature2'}
    )

  @parameterized.parameters(-1, 0, 1)
  def test_raise_error_if_stack_size_is_not_greater_than_1(self, stack_size):
    """Tests that an error is raised if stack_size is not greater than 1."""
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Stack size should be greater than 1.'
    ):
      stack_features_producer.StackFeaturesProducer(
          name='foo', stacked_features_configs=[], stack_size=stack_size
      )


if __name__ == '__main__':
  absltest.main()
