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
from reaf.common import relative_clip_commands_processor

from absl.testing import absltest
from absl.testing import parameterized


class RelativeClipCommandsProcessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'scalar_command',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1]),
          'delta_upper': np.array([0.2]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7])},
          'consumed_commands': {'position': np.array([0.9])},
          'expected_processed_commands': {'position': np.array([0.9])},
      },
      {
          'testcase_name': 'scalar_command_outside_max',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1]),
          'delta_upper': np.array([0.2]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7])},
          'consumed_commands': {'position': np.array([1.0])},
          'expected_processed_commands': {'position': np.array([0.9])},
      },
      {
          'testcase_name': 'scalar_command_outside_min',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1]),
          'delta_upper': np.array([0.2]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7])},
          'consumed_commands': {'position': np.array([0.3])},
          'expected_processed_commands': {'position': np.array([0.6])},
      },
      {
          'testcase_name': 'multidimensional_command',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1, 0.2, 0.3]),
          'delta_upper': np.array([0.2, 0.3, 0.4]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7, 0.6, 0.5])},
          'consumed_commands': {'position': np.array([0.8, 0.5, 0.3])},
          'expected_processed_commands': {
              'position': np.array([0.8, 0.5, 0.3])
          },
      },
      {
          'testcase_name': 'multidimensional_command_different_limits_per_dim',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1, 0.2, 0.3]),
          'delta_upper': np.array([0.2, 0.3, 0.4]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7, 0.6, 0.5])},
          'consumed_commands': {'position': np.array([0.5, 1.0, 0.1])},
          'expected_processed_commands': {
              'position': np.array([0.6, 0.9, 0.2])
          },
      },
  )
  def test_process_commands(
      self,
      command_to_clip: str,
      delta_lower: np.ndarray,
      delta_upper: np.ndarray,
      reference_feature_key: str,
      observed_features: Mapping[str, np.ndarray],
      consumed_commands: Mapping[str, np.ndarray],
      expected_processed_commands: Mapping[str, np.ndarray],
  ):
    processor = relative_clip_commands_processor.RelativeClipCommandsProcessor(
        name='test_relative_clipper',
        command_to_clip=command_to_clip,
        command_spec=specs.BoundedArray(
            shape=delta_lower.shape,
            dtype=delta_lower.dtype,
            minimum=np.zeros(delta_lower.shape),
            maximum=np.ones(delta_lower.shape) * 2,
        ),
        delta_lower=delta_lower,
        delta_upper=delta_upper,
        reference_feature_key=reference_feature_key,
    )
    processor.observe_features(observed_features)
    processed_commands = processor.process_commands(consumed_commands)
    self.assert_numpy_dict_almost_equal(
        processed_commands, expected_processed_commands
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'delta_lower_command_wrong_shape',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1, 0.2]),
          'delta_upper': np.array([0.2]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7])},
          'consumed_commands': {'position': np.array([0.9])},
      },
      {
          'testcase_name': 'delta_upper_command_wrong_shape',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1]),
          'delta_upper': np.array([0.2, 0.3]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7])},
          'consumed_commands': {'position': np.array([0.9])},
      },
      {
          'testcase_name': 'delta_observed_command_wrong_shape',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1]),
          'delta_upper': np.array([0.2]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7, 0.8])},
          'consumed_commands': {'position': np.array([0.9])},
      },
      {
          'testcase_name': 'delta_consumed_command_wrong_shape',
          'command_to_clip': 'position',
          'delta_lower': np.array([0.1]),
          'delta_upper': np.array([0.2]),
          'reference_feature_key': 'observed_position',
          'observed_features': {'observed_position': np.array([0.7])},
          'consumed_commands': {'position': np.array([0.9, 0.8])},
      },
  )
  def test_raises_if_shape_of_arguments_is_incorrect(
      self,
      command_to_clip,
      delta_lower,
      delta_upper,
      reference_feature_key,
      observed_features,
      consumed_commands,
  ):
    with self.assertRaisesRegex(ValueError, '([Ss]hape).*(match)'):
      processor = (
          relative_clip_commands_processor.RelativeClipCommandsProcessor(
              name='test_clipper',
              command_to_clip=command_to_clip,
              command_spec=specs.BoundedArray(
                  shape=(1,),
                  dtype=np.float64,
                  minimum=np.array([0.0]),
                  maximum=np.array([2.0]),
              ),
              delta_lower=delta_lower,
              delta_upper=delta_upper,
              reference_feature_key=reference_feature_key,
          )
      )
      processor.observe_features(observed_features)
      processor.process_commands(consumed_commands)

  def test_produced_commands_keys(self):
    processor = relative_clip_commands_processor.RelativeClipCommandsProcessor(
        name='test_clipper',
        command_to_clip='position',
        command_spec=specs.BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=np.array([0.0]),
            maximum=np.array([2.0]),
        ),
        delta_lower=np.array([0.1]),
        delta_upper=np.array([0.2]),
        reference_feature_key='observed_position',
    )
    self.assertEqual(processor.produced_commands_keys(), {'position'})

  def test_name(self):
    processor = relative_clip_commands_processor.RelativeClipCommandsProcessor(
        name='test_clipper',
        command_to_clip='position',
        command_spec=specs.BoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=np.array([0.0]),
            maximum=np.array([2.0]),
        ),
        delta_lower=np.array([0.1]),
        delta_upper=np.array([0.2]),
        reference_feature_key='observed_position',
    )
    self.assertEqual(processor.name, 'test_clipper')

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
