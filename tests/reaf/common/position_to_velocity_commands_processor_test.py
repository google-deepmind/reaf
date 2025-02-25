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
from reaf.common import position_to_velocity_commands_processor

from absl.testing import absltest
from absl.testing import parameterized


class PositionToVelocityCommandsProcessorTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'zero_error_zero_velocity',
          'desired_positions': np.array(1.0),
          'current_positions': np.array(1.0),
          'current_velocity_command': np.array(0.0),
          'minimum_velocity_command': np.array(-1.0),
          'maximum_velocity_command': np.array(1.0),
          'feedback_gains': np.array(1.0),
          'max_acceleration': np.array(1.0),
          'expected_velocity_command': np.array(0.0),
      },
      {
          'testcase_name': 'positive_error_positive_velocity',
          'desired_positions': np.array(1.5),
          'current_positions': np.array(1.0),
          'current_velocity_command': np.array(0.0),
          'minimum_velocity_command': np.array(-1.0),
          'maximum_velocity_command': np.array(1.0),
          'feedback_gains': np.array(1.0),
          'max_acceleration': np.array(1.0),
          'expected_velocity_command': np.array(0.5),
      },
      {
          'testcase_name': 'negative_error_negative_velocity',
          'desired_positions': np.array(0.5),
          'current_positions': np.array(1.0),
          'current_velocity_command': np.array(0.0),
          'minimum_velocity_command': np.array(-1.0),
          'maximum_velocity_command': np.array(1.0),
          'feedback_gains': np.array(1.0),
          'max_acceleration': np.array(1.0),
          'expected_velocity_command': np.array(-0.5),
      },
      {
          'testcase_name': 'multiple_joints',
          'desired_positions': np.array([1.5, 0.5, 1.2]),
          'current_positions': np.array([1.0, 1.0, 1.0]),
          'current_velocity_command': np.array([0.0, 0.0, 0.0]),
          'minimum_velocity_command': np.array([-1.0, -1.0, -1.0]),
          'maximum_velocity_command': np.array([1.0, 1.0, 1.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([1.0, 1.0, 1.0]),
          'expected_velocity_command': np.array([0.5, -0.5, 0.2]),
      },
      {
          'testcase_name': 'different_feedback_gains',
          'desired_positions': np.array([1.5, 0.5, 1.2]),
          'current_positions': np.array([1.0, 1.0, 1.0]),
          'current_velocity_command': np.array([0.0, 0.0, 0.0]),
          'minimum_velocity_command': np.array([-1.0, -1.0, -1.0]),
          'maximum_velocity_command': np.array([1.0, 1.0, 1.0]),
          'feedback_gains': np.array([2.0, 0.5, 1.5]),
          'max_acceleration': np.array([1.0, 1.0, 1.0]),
          'expected_velocity_command': np.array([1.0, -0.25, 0.3]),
      },
      {
          'testcase_name': 'clipped_velocity_delta',
          'desired_positions': np.array([3.8, -0.2, 2.4]),
          'current_positions': np.array([1.0, 1.0, 1.0]),
          'current_velocity_command': np.array([0.0, 0.0, 0.0]),
          'minimum_velocity_command': np.array([-10.0, -10.0, -10.0]),
          'maximum_velocity_command': np.array([10.0, 10.0, 10.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([0.5, 0.5, 0.5]),
          'expected_velocity_command': np.array([0.5, -0.5, 0.5]),
      },
      {
          'testcase_name': 'clipped_final_velocity',
          'desired_positions': np.array([2.0, -1.0, 2.5]),
          'current_positions': np.array([1.0, 1.0, 1.0]),
          'current_velocity_command': np.array([0.0, 0.0, 0.0]),
          'minimum_velocity_command': np.array([-1.0, -1.0, -1.0]),
          'maximum_velocity_command': np.array([1.0, 1.0, 1.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([np.inf, np.inf, np.inf]),
          'expected_velocity_command': np.array([1.0, -1.0, 1.0]),
      },
  )
  def test_compute_velocity_command(
      self,
      desired_positions: np.ndarray,
      current_positions: np.ndarray,
      current_velocity_command: np.ndarray,
      minimum_velocity_command: np.ndarray,
      maximum_velocity_command: np.ndarray,
      feedback_gains: np.ndarray,
      max_acceleration: np.ndarray,
      expected_velocity_command: np.ndarray,
  ):
    velocity_command = (
        position_to_velocity_commands_processor._compute_velocity_command(
            desired_positions=desired_positions,
            current_positions=current_positions,
            current_velocity_command=current_velocity_command,
            minimum_velocity_command=minimum_velocity_command,
            maximum_velocity_command=maximum_velocity_command,
            feedback_gains=feedback_gains,
            max_acceleration=max_acceleration,
            control_timestep=1.0,
        )
    )
    np.testing.assert_array_almost_equal(
        velocity_command, expected_velocity_command
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'scalar_command',
          'consumed_position_command_key': 'desired_position',
          'produced_velocity_command_key': 'velocity_command',
          'position_reference_feature_key': 'current_position',
          'consumed_position_command_spec': specs.BoundedArray(
              shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
          ),
          'current_position': np.array(1.0),
          'consumed_commands': {'desired_position': np.array(1.5)},
          'minimum_velocity_command': np.array(-1.0),
          'maximum_velocity_command': np.array(1.0),
          'feedback_gains': np.array(1.0),
          'max_acceleration': np.array(1.0),
          'expected_processed_commands': {'velocity_command': np.array(0.5)},
      },
      {
          'testcase_name': 'multidimensional_command',
          'consumed_position_command_key': 'desired_positions',
          'produced_velocity_command_key': 'velocity_commands',
          'position_reference_feature_key': 'current_positions',
          'consumed_position_command_spec': specs.BoundedArray(
              shape=(3,),
              dtype=np.float64,
              minimum=[0.0, 0.0, 0.0],
              maximum=[2.0, 2.0, 2.0],
          ),
          'current_position': np.array([0.5, 1.0, 1.5]),
          'consumed_commands': {'desired_positions': np.array([1.0, 1.5, 2.0])},
          'minimum_velocity_command': np.array([-1.0, -1.0, -1.0]),
          'maximum_velocity_command': np.array([1.0, 1.0, 1.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([1.0, 1.0, 1.0]),
          'expected_processed_commands': {
              'velocity_commands': np.array([0.5, 0.5, 0.5])
          },
      },
      {
          'testcase_name': 'clipped_velocity_command',
          'consumed_position_command_key': 'desired_positions',
          'produced_velocity_command_key': 'velocity_commands',
          'position_reference_feature_key': 'current_positions',
          'consumed_position_command_spec': specs.BoundedArray(
              shape=(3,),
              dtype=np.float64,
              minimum=[-100.0, -100.0, -100.0],
              maximum=[100.0, 100.0, 100.0],
          ),
          'current_position': np.array([0.5, 1.0, 1.5]),
          'consumed_commands': {
              'desired_positions': np.array([10.0, -100.0, 7.0])
          },
          'minimum_velocity_command': np.array([-1.0, -3.0, -4.0]),
          'maximum_velocity_command': np.array([1.0, 3.0, 4.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([np.inf, np.inf, np.inf]),
          'expected_processed_commands': {
              'velocity_commands': np.array([1.0, -3.0, 4.0])
          },
      },
  )
  def test_process_commands(
      self,
      consumed_position_command_key: str,
      produced_velocity_command_key: str,
      position_reference_feature_key: str,
      consumed_position_command_spec: specs.BoundedArray,
      current_position: np.ndarray,
      consumed_commands: Mapping[str, np.ndarray],
      minimum_velocity_command: np.ndarray,
      maximum_velocity_command: np.ndarray,
      feedback_gains: np.ndarray,
      max_acceleration: np.ndarray,
      expected_processed_commands: Mapping[str, np.ndarray],
  ):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key=consumed_position_command_key,
            produced_velocity_command_key=produced_velocity_command_key,
            position_reference_feature_key=position_reference_feature_key,
            consumed_position_command_spec=consumed_position_command_spec,
            minimum_velocity_command=minimum_velocity_command,
            maximum_velocity_command=maximum_velocity_command,
            feedback_gains=feedback_gains,
            max_acceleration=max_acceleration,
            control_timestep=1.0,
        ),
    )
    # We need to call observe_features once to set the current position.
    processor.observe_features(
        {position_reference_feature_key: current_position}
    )
    self.assert_numpy_dict_almost_equal(
        processor.process_commands(consumed_commands),
        expected_processed_commands,
    )

  def test_consumed_commands_spec(self):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=specs.BoundedArray(
                shape=(2,),
                dtype=np.float64,
                minimum=[0.0, 0.0],
                maximum=[2.0, 2.0],
            ),
            minimum_velocity_command=np.array([-1.0, -1.0]),
            maximum_velocity_command=np.array([1.0, 1.0]),
            feedback_gains=np.array([1.0, 1.0]),
            max_acceleration=np.array([1.0, 1.0]),
            control_timestep=1.0,
        ),
    )
    self.assertDictEqual(
        processor.consumed_commands_spec(),
        {
            'desired_position': specs.BoundedArray(
                shape=(2,),
                dtype=np.float64,
                minimum=[0.0, 0.0],
                maximum=[2.0, 2.0],
            )
        },
    )

  def test_produced_commands_keys(self):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=specs.BoundedArray(
                shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
            ),
            minimum_velocity_command=np.array(-1.0),
            maximum_velocity_command=np.array(1.0),
            feedback_gains=np.array(1.0),
            max_acceleration=np.array(1.0),
            control_timestep=1.0,
        ),
    )
    self.assertEqual(processor.produced_commands_keys(), {'velocity_command'})

  def test_name(self):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=specs.BoundedArray(
                shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
            ),
            minimum_velocity_command=np.array(-1.0),
            maximum_velocity_command=np.array(1.0),
            feedback_gains=np.array(1.0),
            max_acceleration=np.array(1.0),
            control_timestep=1.0,
        ),
    )
    self.assertEqual(processor.name, 'test_position_to_velocity_processor')

  def test_raises_if_reference_feature_shape_is_wrong(self):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=specs.BoundedArray(
                shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
            ),
            minimum_velocity_command=np.array(-1.0),
            maximum_velocity_command=np.array(1.0),
            feedback_gains=np.array(1.0),
            max_acceleration=np.array(1.0),
            control_timestep=1.0,
        ),
    )
    with self.assertRaisesRegex(ValueError, 'Shape of reference feature'):
      processor.observe_features({'current_position': np.array([1.0, 1.0])})

  def test_raises_if_reference_feature_dtype_is_wrong(self):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=specs.BoundedArray(
                shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
            ),
            minimum_velocity_command=np.array(-1.0),
            maximum_velocity_command=np.array(1.0),
            feedback_gains=np.array(1.0),
            max_acceleration=np.array(1.0),
            control_timestep=1.0,
        ),
    )
    with self.assertRaisesRegex(ValueError, 'Dtype of reference feature'):
      processor.observe_features(
          {'current_position': np.array(1.0, dtype=np.int64)}
      )

  @parameterized.named_parameters(
      {
          'testcase_name': 'wrong_dtype',
          'spec': specs.BoundedArray(
              shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
          ),
          'features': {'current_position': np.array(1.0, dtype=np.int64)},
      },
      {
          'testcase_name': 'wrong_shape',
          'spec': specs.BoundedArray(
              shape=(), dtype=np.float64, minimum=0.0, maximum=2.0
          ),
          'features': {'current_position': np.array([1.0, 1.0])},
      },
  )
  def test_does_not_raise_if_validation_is_disabled(
      self, spec: specs.BoundedArray, features: Mapping[str, np.ndarray]
  ):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=spec,
            minimum_velocity_command=np.array(-1.0),
            maximum_velocity_command=np.array(1.0),
            feedback_gains=np.array(1.0),
            max_acceleration=np.array(1.0),
            control_timestep=1.0,
        ),
        validate_reference_feature=False,
    )
    processor.observe_features(features)

  def test_does_not_raise_if_reference_feature_is_wrong_after_successful_validation(
      self,
  ):
    processor = position_to_velocity_commands_processor.PositionToVelocityCommandsProcessor(
        name='test_position_to_velocity_processor',
        config=position_to_velocity_commands_processor.PositionToVelocityConfig(
            consumed_position_command_key='desired_position',
            produced_velocity_command_key='velocity_command',
            position_reference_feature_key='current_position',
            consumed_position_command_spec=specs.BoundedArray(
                shape=(),
                dtype=np.float64,
                minimum=0.0,
                maximum=2.0,
            ),
            minimum_velocity_command=np.array(-1.0),
            maximum_velocity_command=np.array(1.0),
            feedback_gains=np.array(1.0),
            max_acceleration=np.array(1.0),
            control_timestep=1.0,
        ),
    )
    # First, observe a valid feature.
    processor.observe_features({'current_position': np.array(1.0)})
    # Then, observe an invalid feature. This should not raise an error.
    processor.observe_features({'current_position': np.array([1.0, 1.0])})

    # Check that the internal state contains the wrong array.
    np.testing.assert_array_equal(
        processor._current_position, np.array([1.0, 1.0])
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'control_timestep_0.1',
          'control_timestep': 0.1,
          'desired_positions': np.array([1.5, 0.5, 1.2]),
          'current_positions': np.array([1.0, 1.0, 1.0]),
          'current_velocity_command': np.array([0.0, 0.0, 0.0]),
          'minimum_velocity_command': np.array([-1.0, -1.0, -1.0]),
          'maximum_velocity_command': np.array([1.0, 1.0, 1.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([1.0, 1.0, 1.0]),
          'expected_velocity_command': np.array([0.1, -0.1, 0.1]),
      },
      {
          'testcase_name': 'control_timestep_0.5',
          'control_timestep': 0.5,
          'desired_positions': np.array([1.5, 0.5, 1.2]),
          'current_positions': np.array([1.0, 1.0, 1.0]),
          'current_velocity_command': np.array([0.0, 0.0, 0.0]),
          'minimum_velocity_command': np.array([-1.0, -1.0, -1.0]),
          'maximum_velocity_command': np.array([1.0, 1.0, 1.0]),
          'feedback_gains': np.array([1.0, 1.0, 1.0]),
          'max_acceleration': np.array([0.3, 0.1, 1.0]),
          'expected_velocity_command': np.array([0.15, -0.05, 0.2]),
      },
  )
  def test_control_timestep(
      self,
      control_timestep: float,
      desired_positions: np.ndarray,
      current_positions: np.ndarray,
      current_velocity_command: np.ndarray,
      minimum_velocity_command: np.ndarray,
      maximum_velocity_command: np.ndarray,
      feedback_gains: np.ndarray,
      max_acceleration: np.ndarray,
      expected_velocity_command: np.ndarray,
  ):
    velocity_command = (
        position_to_velocity_commands_processor._compute_velocity_command(
            desired_positions=desired_positions,
            current_positions=current_positions,
            current_velocity_command=current_velocity_command,
            minimum_velocity_command=minimum_velocity_command,
            maximum_velocity_command=maximum_velocity_command,
            feedback_gains=feedback_gains,
            max_acceleration=max_acceleration,
            control_timestep=control_timestep,
        )
    )
    np.testing.assert_array_almost_equal(
        velocity_command, expected_velocity_command
    )

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
