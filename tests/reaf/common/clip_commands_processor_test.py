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
from reaf.common import clip_commands_processor

from absl.testing import absltest
from absl.testing import parameterized


class ClipCommandsProcessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'scalar_command',
          'command_to_clip': 'force',
          'min_command': np.array([0.5]),
          'max_command': np.array([1.5]),
          'consumed_commands': {'force': np.array([0.7])},
          'expected_processed_commands': {'force': np.array([0.7])},
      },
      {
          'testcase_name': 'scalar_command_outside_max',
          'command_to_clip': 'force',
          'min_command': np.array([0.5]),
          'max_command': np.array([1.5]),
          'consumed_commands': {'force': np.array([2.0])},
          'expected_processed_commands': {'force': np.array([1.5])},
      },
      {
          'testcase_name': 'scalar_command_outside_min',
          'command_to_clip': 'force',
          'min_command': np.array([0.5]),
          'max_command': np.array([1.5]),
          'consumed_commands': {'force': np.array([0.1])},
          'expected_processed_commands': {'force': np.array([0.5])},
      },
      {
          'testcase_name': 'multidimensional_command',
          'command_to_clip': 'velocity',
          'min_command': np.array([0.0, 0.0, 0.0]),
          'max_command': np.array([1.0, 1.0, 1.0]),
          'consumed_commands': {'velocity': np.array([0.3, 0.7, 1.2])},
          'expected_processed_commands': {
              'velocity': np.array([0.3, 0.7, 1.0])
          },
      },
      {
          'testcase_name': 'multidimensional_command_different_limits_per_dim',
          'command_to_clip': 'velocity',
          'min_command': np.array([0.0, 0.2, 0.5]),
          'max_command': np.array([1.0, 0.7, 1.5]),
          'consumed_commands': {'velocity': np.array([0.3, 0.8, 0.4])},
          'expected_processed_commands': {
              'velocity': np.array([0.3, 0.7, 0.5])
          },
      },
  )
  def test_process_commands(
      self,
      command_to_clip: str,
      min_command: np.ndarray,
      max_command: np.ndarray,
      consumed_commands: Mapping[str, np.ndarray],
      expected_processed_commands: Mapping[str, np.ndarray],
  ):
    processor = clip_commands_processor.ClipCommandsProcessor(
        name='test_clipper',
        command_to_clip=command_to_clip,
        min_command=min_command,
        max_command=max_command,
    )
    processed_commands = processor.process_commands(consumed_commands)
    self.assert_numpy_dict_almost_equal(
        processed_commands, expected_processed_commands
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'min_command_wrong_shape',
          'command_to_clip': 'force',
          'min_command': np.array([0.5, 0.5]),
          'max_command': np.array([1.5]),
      },
      {
          'testcase_name': 'max_command_wrong_shape',
          'command_to_clip': 'force',
          'min_command': np.array([0.5]),
          'max_command': np.array([1.5, 1.5]),
      },
  )
  def test_raises_if_shape_of_arguments_is_incorrect(
      self,
      command_to_clip,
      min_command,
      max_command,
  ):
    with self.assertRaisesRegex(ValueError, 'shape .* not match'):
      clip_commands_processor.ClipCommandsProcessor(
          name='test_clipper',
          command_to_clip=command_to_clip,
          min_command=min_command,
          max_command=max_command,
      )

  def test_process_commands_without_relative_range_clip(self):
    processor = clip_commands_processor.ClipCommandsProcessor(
        name='test_clipper',
        command_to_clip='force',
        min_command=np.array([0.5]),
        max_command=np.array([1.5]),
    )
    processed_commands = processor.process_commands({'force': np.array([0.7])})
    self.assert_numpy_dict_almost_equal(
        processed_commands, {'force': np.array([0.7])}
    )
    processed_commands = processor.process_commands({'force': np.array([2.0])})
    self.assert_numpy_dict_almost_equal(
        processed_commands, {'force': np.array([1.5])}
    )

  def test_consumed_commands_spec(self):
    processor = clip_commands_processor.ClipCommandsProcessor(
        name='test_clipper',
        command_to_clip='force',
        min_command=np.array([0.5, 0.2]),
        max_command=np.array([1.5, 1.8]),
    )
    self.assertDictEqual(
        processor.consumed_commands_spec(),
        {
            'force': specs.BoundedArray(
                shape=(2,),
                dtype=np.float64,
                minimum=[0.5, 0.2],
                maximum=[1.5, 1.8],
            )
        },
    )

  def test_produced_commands_keys(self):
    processor = clip_commands_processor.ClipCommandsProcessor(
        name='test_clipper',
        command_to_clip='force',
        min_command=np.array([0.5]),
        max_command=np.array([1.5]),
    )
    self.assertEqual(processor.produced_commands_keys(), {'force'})

  def test_name(self):
    processor = clip_commands_processor.ClipCommandsProcessor(
        name='test_clipper',
        command_to_clip='force',
        min_command=np.array([0.5]),
        max_command=np.array([1.5]),
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
