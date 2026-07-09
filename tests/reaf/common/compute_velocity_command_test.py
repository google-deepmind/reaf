# Copyright 2026 Google LLC
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
import numpy as np
from reaf.common import compute_velocity_command
from absl.testing import absltest
from absl.testing import parameterized


class ComputeVelocityCommandTest(parameterized.TestCase):

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
        compute_velocity_command.compute_velocity_command(
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
        compute_velocity_command.compute_velocity_command(
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


if __name__ == "__main__":
  absltest.main()
