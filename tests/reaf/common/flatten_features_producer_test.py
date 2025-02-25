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
"""Test for flatten features producer."""

from dm_env import specs
import numpy as np
from reaf.common import flatten_features_producer
from absl.testing import absltest


class FlattenFeaturesProducerTest(absltest.TestCase):

  def test_name(self):
    producer = flatten_features_producer.FlattenFeaturesProducer('test', [])
    self.assertEqual(producer.name, 'test')

  def test_produce_features(self):
    producer = flatten_features_producer.FlattenFeaturesProducer(
        'test',
        [
            flatten_features_producer.FlattenedFeatureConfig(
                feature_name='feature1',
                flattened_feature_name='feature1_flattened',
                flattened_size=6,
                dtype=np.int32,
            )
        ],
    )
    features = {'feature1': np.array([[1, 2, 3], [4, 5, 6]], np.int32)}
    produced_features = producer.produce_features(features)
    np.testing.assert_array_equal(
        produced_features['feature1_flattened'], np.array([1, 2, 3, 4, 5, 6])
    )

  def test_produce_features_column_major(self):
    producer = flatten_features_producer.FlattenFeaturesProducer(
        'test',
        [
            flatten_features_producer.FlattenedFeatureConfig(
                feature_name='feature1',
                flattened_feature_name='feature1_flattened',
                flattened_size=6,
                dtype=np.int32,
                order=flatten_features_producer.Order.COLUMN_MAJOR,
            )
        ],
    )
    features = {'feature1': np.array([[1, 2, 3], [4, 5, 6]], np.int32)}
    produced_features = producer.produce_features(features)
    np.testing.assert_array_equal(
        produced_features['feature1_flattened'], np.array([1, 4, 2, 5, 3, 6])
    )

  def test_produced_features_spec(self):
    producer = flatten_features_producer.FlattenFeaturesProducer(
        'test',
        [
            flatten_features_producer.FlattenedFeatureConfig(
                feature_name='feature1',
                flattened_feature_name='feature1_flattened',
                flattened_size=6,
                dtype=np.int32,
            )
        ],
    )
    spec = producer.produced_features_spec()
    self.assertEqual(
        spec['feature1_flattened'], specs.Array(shape=(6,), dtype=np.int32)
    )

  def test_required_features_keys(self):
    producer = flatten_features_producer.FlattenFeaturesProducer(
        'test',
        [
            flatten_features_producer.FlattenedFeatureConfig(
                feature_name='feature1',
                flattened_feature_name='feature1_flattened',
                flattened_size=6,
                dtype=np.int32,
            ),
            flatten_features_producer.FlattenedFeatureConfig(
                feature_name='feature2',
                flattened_feature_name='feature2_flattened',
                flattened_size=6,
                dtype=np.int32,
            ),
        ],
    )
    self.assertEqual(
        producer.required_features_keys(), {'feature1', 'feature2'}
    )


if __name__ == '__main__':
  absltest.main()
