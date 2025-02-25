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
from collections.abc import Mapping

from dm_env import specs
import numpy as np
from reaf.common import delta_to_absolute_commands_processor

from absl.testing import absltest
from absl.testing import parameterized


class DeltaToAbsoluteCommandsProcessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'scalar_command',
          'consumed_command_key': 'force_delta',
          'produced_command_key': 'force',
          'reference_feature_key': 'force_sensor',
          'consumed_command_spec': specs.BoundedArray(
              shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
          ),
          'reference_value': np.array(1.0),
          'consumed_commands': {'force_delta': np.array(0.7)},
          'expected_processed_commands': {'force': np.array(1.7)},
      },
      {
          'testcase_name': 'multidimensional_command',
          'consumed_command_key': 'velocity_delta',
          'produced_command_key': 'velocity',
          'reference_feature_key': 'velocity_sensor',
          'consumed_command_spec': specs.BoundedArray(
              shape=(3,),
              dtype=np.float64,
              minimum=[-1.0, -1.0, -1.0],
              maximum=[1.0, 1.0, 1.0],
          ),
          'reference_value': np.array([0.5, 0.5, 0.9]),
          'consumed_commands': {'velocity_delta': np.array([0.3, 0.7, -0.2])},
          'expected_processed_commands': {
              'velocity': np.array([0.8, 1.2, 0.7])
          },
      },
  )
  def test_process_commands(
      self,
      consumed_command_key: str,
      produced_command_key: str,
      reference_feature_key: str,
      consumed_command_spec: specs.BoundedArray,
      reference_value: np.ndarray,
      consumed_commands: Mapping[str, np.ndarray],
      expected_processed_commands: Mapping[str, np.ndarray],
  ):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key=consumed_command_key,
                produced_command_key=produced_command_key,
                reference_feature_key=reference_feature_key,
                consumed_command_spec=consumed_command_spec,
            ),
        )
    )
    # We need to call observe_features once to set the reference value.
    processor.observe_features({reference_feature_key: reference_value})
    self.assert_numpy_dict_almost_equal(
        processor.process_commands(consumed_commands),
        expected_processed_commands,
    )

  def test_consumed_commands_spec(self):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key='force_delta',
                produced_command_key='force',
                reference_feature_key='force_sensor',
                consumed_command_spec=specs.BoundedArray(
                    shape=(2,),
                    dtype=np.float64,
                    minimum=[-1.0, -1.0],
                    maximum=[1.0, 1.0],
                ),
            ),
        )
    )
    self.assertDictEqual(
        processor.consumed_commands_spec(),
        {
            'force_delta': specs.BoundedArray(
                shape=(2,),
                dtype=np.float64,
                minimum=[-1.0, -1.0],
                maximum=[1.0, 1.0],
            )
        },
    )

  def test_produced_commands_keys(self):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key='force_delta',
                produced_command_key='force',
                reference_feature_key='force_sensor',
                consumed_command_spec=specs.BoundedArray(
                    shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
                ),
            ),
        )
    )
    self.assertEqual(processor.produced_commands_keys(), {'force'})

  def test_name(self):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key='force_delta',
                produced_command_key='force',
                reference_feature_key='force_sensor',
                consumed_command_spec=specs.BoundedArray(
                    shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
                ),
            ),
        )
    )
    self.assertEqual(processor.name, 'test_delta_processor')

  def test_raises_if_reference_feature_shape_is_wrong(self):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key='force_delta',
                produced_command_key='force',
                reference_feature_key='force_sensor',
                consumed_command_spec=specs.BoundedArray(
                    shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
                ),
            ),
        )
    )
    with self.assertRaisesRegex(ValueError, 'Shape of reference feature'):
      processor.observe_features({'force_sensor': np.array([1.0, 1.0])})

  def test_raises_if_reference_feature_dtype_is_wrong(self):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key='force_delta',
                produced_command_key='force',
                reference_feature_key='force_sensor',
                consumed_command_spec=specs.BoundedArray(
                    shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
                ),
            ),
        )
    )
    with self.assertRaisesRegex(ValueError, 'Dtype of reference feature'):
      processor.observe_features(
          {'force_sensor': np.array(1.0, dtype=np.int64)}
      )

  @parameterized.named_parameters(
      {
          'testcase_name': 'wrong_dtype',
          'spec': specs.BoundedArray(
              shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
          ),
          'features': {'force_sensor': np.array(1.0, dtype=np.int64)},
      },
      {
          'testcase_name': 'wrong_shape',
          'spec': specs.BoundedArray(
              shape=(), dtype=np.float64, minimum=-1.0, maximum=1.0
          ),
          'features': {'force_sensor': np.array([1.0, 1.0])},
      },
  )
  def test_does_not_raise_if_validation_is_disabled(
      self, spec: specs.BoundedArray, features: Mapping[str, np.ndarray]
  ):
    processor = (
        delta_to_absolute_commands_processor.DeltaToAbsoluteCommandsProcessor(
            name='test_delta_processor',
            config=delta_to_absolute_commands_processor.DeltaToAbsoluteConfig(
                consumed_command_key='force_delta',
                produced_command_key='force',
                reference_feature_key='force_sensor',
                consumed_command_spec=spec,
            ),
            validate_reference_feature=False,
        )
    )
    processor.observe_features(features)

  def assert_numpy_dict_almost_equal(
      self, a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]
  ):
    """Asserts that two numpy dictionaries are equal."""
    self.assertLen(a, len(b))
    for key in a:
      self.assertIn(key, b)
      np.testing.assert_array_almost_equal(a[key], b[key])


if __name__ == '__main__':
  absltest.main()
