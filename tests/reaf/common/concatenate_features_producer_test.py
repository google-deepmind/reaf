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
from reaf.common import concatenate_features_producer
from absl.testing import absltest
from absl.testing import parameterized


class ConcatenateFeaturesProducersTest(parameterized.TestCase):

  def test_name(self):
    producer_name = 'test_producer'
    producer = concatenate_features_producer.ConcatenateFeaturesProducer(
        name=producer_name,
        concatenated_features_specs={
            'feature1': specs.Array(shape=(3,), dtype=np.float32),
            'feature2': specs.Array(shape=(3,), dtype=np.float32),
        },
        new_feature_name='feature3',
    )
    self.assertEqual(producer.name, producer_name)

  def test_produce_features_basic(self):
    features = {
        'feature1': np.array([1, 2, 3]),
        'feature2': np.array([4, 5, 6]),
    }
    producer = concatenate_features_producer.ConcatenateFeaturesProducer(
        name='test',
        concatenated_features_specs={
            'feature1': specs.Array(shape=(3,), dtype=np.float32),
            'feature2': specs.Array(shape=(3,), dtype=np.float32),
        },
        new_feature_name='feature3',
    )
    produced_features = producer.produce_features(features)
    expected_features = {'feature3': np.array([1, 2, 3, 4, 5, 6])}
    self.assertSetEqual(
        set(produced_features.keys()), set(expected_features.keys())
    )
    for key in produced_features.keys():
      np.testing.assert_array_equal(
          produced_features[key], expected_features[key]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='axis_1',
          axis=1,
          expected_features={
              'feature3': np.array([[1, 2, 5, 6], [3, 4, 7, 8]])
          },
      ),
      dict(
          testcase_name='axis_None',
          axis=None,
          expected_features={'feature3': np.array([1, 2, 3, 4, 5, 6, 7, 8])},
      ),
  )
  def test_produce_features_different_axis(self, axis, expected_features):
    features = {
        'feature1': np.array([[1, 2], [3, 4]]),
        'feature2': np.array([[5, 6], [7, 8]]),
    }
    producer = concatenate_features_producer.ConcatenateFeaturesProducer(
        name='test',
        concatenated_features_specs={
            'feature1': specs.Array(shape=(2, 2), dtype=np.float32),
            'feature2': specs.Array(shape=(2, 2), dtype=np.float32),
        },
        new_feature_name='feature3',
        axis=axis,
    )
    produced_features = producer.produce_features(features)
    self.assertSetEqual(
        set(produced_features.keys()), set(expected_features.keys())
    )
    for key in produced_features.keys():
      np.testing.assert_array_equal(
          produced_features[key], expected_features[key]
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='axis_1',
          axis=1,
          expected_spec={
              'feature3': specs.Array(shape=(3, 6), dtype=np.float32)
          },
      ),
      dict(
          testcase_name='axis_None',
          axis=None,
          expected_spec={
              'feature3': specs.Array(shape=(18,), dtype=np.float32)
          },
      ),
  )
  def test_produced_features_spec(self, axis, expected_spec):
    producer = concatenate_features_producer.ConcatenateFeaturesProducer(
        name='test',
        concatenated_features_specs={
            'feature1': specs.Array(shape=(3, 2), dtype=np.float32),
            'feature2': specs.Array(shape=(3, 4), dtype=np.float32),
        },
        new_feature_name='feature3',
        axis=axis,
    )
    produced_spec = producer.produced_features_spec()
    self.assertEqual(produced_spec, expected_spec)

  def test_required_features_keys(self):
    producer = concatenate_features_producer.ConcatenateFeaturesProducer(
        name='test',
        concatenated_features_specs={
            'feature1': specs.Array(shape=(3,), dtype=np.float32),
            'feature2': specs.Array(shape=(3,), dtype=np.float32),
        },
        new_feature_name='feature3',
    )
    required_keys = producer.required_features_keys()
    self.assertEqual(required_keys, {'feature1', 'feature2'})

  def test_produced_features_spec_different_dtypes_raises_error(self):

    with self.assertRaises(TypeError):
      concatenate_features_producer.ConcatenateFeaturesProducer(
          name='test',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(3,), dtype=np.float32),
              'feature2': specs.Array(shape=(3,), dtype=np.int32),
          },
          new_feature_name='feature3',
      )

  def test_produced_features_spec_shape_mismatch_raises_error_when_axis_not_none(
      self,
  ):
    with self.assertRaises(ValueError):
      concatenate_features_producer.ConcatenateFeaturesProducer(
          name='test',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(2, 3), dtype=np.float32),
              # Different second dimension
              'feature2': specs.Array(shape=(3, 4), dtype=np.float32),
          },
          new_feature_name='feature3',
          axis=0,
      )

  def test_produced_features_spec_shape_mismatch_when_axis_is_none(self):
    producer = concatenate_features_producer.ConcatenateFeaturesProducer(
        name='test',
        concatenated_features_specs={
            'feature1': specs.Array(shape=(2, 3), dtype=np.float32),
            # Different second dimension
            'feature2': specs.Array(shape=(3, 4), dtype=np.float32),
        },
        new_feature_name='feature3',
        axis=None,
    )
    expected_spec = {'feature3': specs.Array(shape=(18,), dtype=np.float32)}
    produced_spec = producer.produced_features_spec()
    self.assertEqual(produced_spec, expected_spec)

  def test_produced_features_spec_dimension_number_mismatch_raises_error(self):
    with self.assertRaises(ValueError):
      concatenate_features_producer.ConcatenateFeaturesProducer(
          name='test',
          concatenated_features_specs={
              'feature1': specs.Array(shape=(2, 3, 3), dtype=np.float32),
              # Different second dimension
              'feature2': specs.Array(shape=(2, 3), dtype=np.float32),
          },
          new_feature_name='feature3',
          axis=0,
      )


if __name__ == '__main__':
  absltest.main()
