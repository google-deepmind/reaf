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
from unittest import mock
from dm_env import specs
import numpy as np
from reaf.core import default_observation_space_adapter as obs_adapter
from reaf.core import numpy_mock_assertions
from absl.testing import absltest


class DefaultObservationSpaceAdapterTest(absltest.TestCase):

  def test_bad_max_float_dtype_raises_error(self):
    with self.assertRaises(ValueError):
      obs_adapter.DefaultObservationSpaceAdapter(
          task_features_spec={},
          selected_features=None,
          renamed_features=None,
          observation_type_mapper=None,
          max_float_dtype=int,
      )

  def test_downcasting_success(self):
    task_features_spec = {
        "int32_feature": specs.Array(shape=(2,), dtype=np.int32),
        "int64_feature": specs.Array(shape=(2,), dtype=np.int64),
        "float32_feature": specs.Array(shape=(2,), dtype=np.float32),
        "float64_feature": specs.Array(shape=(2,), dtype=np.float64),
    }
    adapter = obs_adapter.DefaultObservationSpaceAdapter(
        task_features_spec=task_features_spec,
        selected_features=None,
        renamed_features=None,
        observation_type_mapper=None,
        max_float_dtype=np.float32,
    )

    # Only the float64 should have been cast down.
    self.assertEqual(
        adapter.observation_spec(),
        {
            "int32_feature": specs.Array(shape=(2,), dtype=np.int32),
            "int64_feature": specs.Array(shape=(2,), dtype=np.int64),
            "float32_feature": specs.Array(shape=(2,), dtype=np.float32),
            "float64_feature": specs.Array(shape=(2,), dtype=np.float32),
        },
    )

    features = {
        "int32_feature": np.asarray([-1, 1], dtype=np.int32),
        "int64_feature": np.asarray([-2, 2], dtype=np.int64),
        "float32_feature": np.asarray([-1.1, 1.1], dtype=np.float32),
        "float64_feature": np.asarray([-2.2, 2.2], dtype=np.float64),
    }

    # Only the float64 should have been cast down.
    np.testing.assert_equal(
        adapter.observations_from_features(features),
        {
            "int32_feature": np.asarray([-1, 1], dtype=np.int32),
            "int64_feature": np.asarray([-2, 2], dtype=np.int64),
            "float32_feature": np.asarray([-1.1, 1.1], dtype=np.float32),
            "float64_feature": np.asarray([-2.2, 2.2], dtype=np.float32),
        },
    )

  def test_filtering(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    adapter = obs_adapter.DefaultObservationSpaceAdapter(
        task_features_spec=task_features_spec,
        selected_features=("feature1", "feature3"),
        renamed_features=None,
        observation_type_mapper=None,
    )

    self.assertEqual(
        adapter.observation_spec(),
        {
            "feature1": specs.Array(shape=(3,), dtype=np.float32),
            "feature3": specs.Array(shape=(4,), dtype=np.int32),
        },
    )

    features = {
        "feature1": np.asarray([1, 4.5, -23]),
        "feature2": np.asarray([0.1, 0.99]),
        "feature3": np.asarray([1, -2, 3, 5]),
    }

    np.testing.assert_equal(
        adapter.observations_from_features(features),
        {
            "feature1": np.asarray([1, 4.5, -23]),
            "feature3": np.asarray([1, -2, 3, 5]),
        },
    )

  def test_no_filter_converts_all_features(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    adapter = obs_adapter.DefaultObservationSpaceAdapter(
        task_features_spec=task_features_spec,
        selected_features=None,
        renamed_features=(),
        observation_type_mapper=None,
    )

    self.assertEqual(
        adapter.observation_spec(),
        {
            "feature1": specs.Array(shape=(3,), dtype=np.float32),
            "feature2": specs.BoundedArray(
                shape=(2,),
                dtype=np.float32,
                minimum=0.0,
                maximum=1.0,
            ),
            "feature3": specs.Array(shape=(4,), dtype=np.int32),
        },
    )

    features = {
        "feature1": np.asarray([1, 4.5, -23]),
        "feature2": np.asarray([0.1, 0.99]),
        "feature3": np.asarray([1, -2, 3, 5]),
    }

    np.testing.assert_equal(
        adapter.observations_from_features(features),
        {
            "feature1": np.asarray([1, 4.5, -23]),
            "feature2": np.asarray([0.1, 0.99]),
            "feature3": np.asarray([1, -2, 3, 5]),
        },
    )

  def test_empty_filter_means_no_observations(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    adapter = obs_adapter.DefaultObservationSpaceAdapter(
        task_features_spec=task_features_spec,
        selected_features=(),
        renamed_features=(),
        observation_type_mapper=None,
    )

    self.assertEmpty(adapter.observation_spec())

    features = {
        "feature1": np.asarray([1, 4.5, -23]),
        "feature2": np.asarray([0.1, 0.99]),
        "feature3": np.asarray([1, -2, 3, 5]),
    }

    self.assertEmpty(adapter.observations_from_features(features))

  def test_renaming(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    adapter = obs_adapter.DefaultObservationSpaceAdapter(
        task_features_spec=task_features_spec,
        selected_features=None,
        renamed_features=(
            obs_adapter.RenameInfo(
                original_key="feature1", renamed_key="new_feature1"
            ),
            obs_adapter.RenameInfo(
                original_key="feature3", renamed_key="new_feature3"
            ),
        ),
        observation_type_mapper=None,
    )

    self.assertEqual(
        adapter.observation_spec(),
        {
            "new_feature1": specs.Array(shape=(3,), dtype=np.float32),
            "feature2": specs.BoundedArray(
                shape=(2,),
                dtype=np.float32,
                minimum=0.0,
                maximum=1.0,
            ),
            "new_feature3": specs.Array(shape=(4,), dtype=np.int32),
        },
    )

    features = {
        "feature1": np.asarray([1, 4.5, -23]),
        "feature2": np.asarray([0.1, 0.99]),
        "feature3": np.asarray([1, -2, 3, 5]),
    }

    np.testing.assert_equal(
        adapter.observations_from_features(features),
        {
            "new_feature1": np.asarray([1, 4.5, -23]),
            "feature2": np.asarray([0.1, 0.99]),
            "new_feature3": np.asarray([1, -2, 3, 5]),
        },
    )

  def test_type_conversion_is_called(self):
    type_remapper = mock.create_autospec(
        obs_adapter.ObservationTypeMapper, instance=True
    )

    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    adapter = obs_adapter.DefaultObservationSpaceAdapter(
        task_features_spec=task_features_spec,
        selected_features=None,
        renamed_features=(),
        observation_type_mapper=type_remapper,
    )

    features = {
        "feature1": np.asarray([1, 4.5, -23]),
        "feature2": np.asarray([0.1, 0.99]),
        "feature3": np.asarray([1, -2, 3, 5]),
    }

    adapter.observation_spec()
    adapter.observations_from_features(features)

    type_remapper.to_observation_spec.assert_called_once_with(
        task_features_spec
    )
    numpy_mock_assertions.assert_called_once_with(
        type_remapper.to_observations, features
    )

  def test_missing_filter_keys_raise_error(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    with self.assertRaises(ValueError):
      obs_adapter.DefaultObservationSpaceAdapter(
          task_features_spec=task_features_spec,
          selected_features=("feature1", "missing_feature"),
          renamed_features=(),
          observation_type_mapper=None,
      )

  def test_missing_rename_keys_raise_error(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    with self.assertRaises(ValueError):
      obs_adapter.DefaultObservationSpaceAdapter(
          task_features_spec=task_features_spec,
          selected_features=None,
          renamed_features=(
              obs_adapter.RenameInfo(
                  original_key="feature1", renamed_key="new_feature1"
              ),
              obs_adapter.RenameInfo(
                  original_key="missing_feature", renamed_key="new_feature"
              ),
          ),
          observation_type_mapper=None,
      )

  def test_missing_rename_keys_after_filter_raise_error(self):
    task_features_spec = {
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
        "feature2": specs.BoundedArray(
            shape=(2,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
        ),
        "feature3": specs.Array(shape=(4,), dtype=np.int32),
    }

    with self.assertRaises(ValueError):
      obs_adapter.DefaultObservationSpaceAdapter(
          task_features_spec=task_features_spec,
          selected_features=("feature1",),
          renamed_features=(
              obs_adapter.RenameInfo(
                  original_key="feature1", renamed_key="new_feature1"
              ),
              obs_adapter.RenameInfo(
                  original_key="feature2", renamed_key="new_feature2"
              ),
          ),
          observation_type_mapper=None,
      )


if __name__ == "__main__":
  absltest.main()
