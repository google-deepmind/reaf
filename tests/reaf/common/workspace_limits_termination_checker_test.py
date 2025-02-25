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

import numpy as np
from reaf.common import workspace_limits_termination_checker
from reaf.core import termination_checker
from absl.testing import absltest


_TEST_NAME = "test_termination_checker"
_POSITION_KEY = "position_key"
_WORKSPACE_SIZE = 5.0


def get_termination_checker() -> (
    workspace_limits_termination_checker.WorkspaceLimitsTerminationChecker
):
  return workspace_limits_termination_checker.WorkspaceLimitsTerminationChecker(
      name=_TEST_NAME,
      workspace=workspace_limits_termination_checker.create_cubic_workspace(
          size_meters=_WORKSPACE_SIZE
      ),
      tcp_position_key=_POSITION_KEY,
  )


class WorkspaceLimitsTerminationCheckerTest(absltest.TestCase):

  def test_name(self):
    self.assertEqual(get_termination_checker().name, _TEST_NAME)

  def test_check_termination_terminates(self):
    self.assertEqual(
        get_termination_checker().check_termination({
            _POSITION_KEY: np.array([
                _WORKSPACE_SIZE / 2,
                0,
                0,
            ]),
        }),
        termination_checker.TerminationResult.TERMINATE,
    )

  def test_check_termination_does_not_terminate(self):
    self.assertEqual(
        get_termination_checker().check_termination({
            _POSITION_KEY: np.array([
                # Divide by 2 as a 5m side coordsponds to 2.5m from the center.
                _WORKSPACE_SIZE / 2 - 0.001,
                0,
                0,
            ]),
        }),
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

  def test_required_features_keys(self):
    self.assertCountEqual(
        [_POSITION_KEY], get_termination_checker().required_features_keys()
    )

  def test_tolerance_does_not_terminate(self):
    tolerance = 0.1
    checker = workspace_limits_termination_checker.WorkspaceLimitsTerminationChecker(
        name=_TEST_NAME,
        workspace=workspace_limits_termination_checker.create_cubic_workspace(
            size_meters=_WORKSPACE_SIZE
        ),
        tcp_position_key=_POSITION_KEY,
        added_x_tolerance=tolerance,
        added_y_tolerance=tolerance,
        added_z_tolerance=tolerance,
    )
    self.assertEqual(
        checker.check_termination({
            # Need to be just inside the bounds so - 0.0001.
            # Divide by 2 because a 5m side coordsponds to 2.5m from the center.
            _POSITION_KEY: np.array([
                _WORKSPACE_SIZE / 2 + tolerance - 0.0001,
                _WORKSPACE_SIZE / 2 + tolerance - 0.0001,
                _WORKSPACE_SIZE / 2 + tolerance - 0.0001,
            ]),
        }),
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

  def test_tolerance_terminate(self):
    tolerance = 0.1
    checker = workspace_limits_termination_checker.WorkspaceLimitsTerminationChecker(
        name=_TEST_NAME,
        workspace=workspace_limits_termination_checker.create_cubic_workspace(
            size_meters=_WORKSPACE_SIZE
        ),
        tcp_position_key=_POSITION_KEY,
        added_x_tolerance=tolerance,
        added_y_tolerance=tolerance,
        added_z_tolerance=tolerance,
    )
    self.assertEqual(
        checker.check_termination({
            # Divide by 2 because a 5m side coordsponds to 2.5m from the center.
            _POSITION_KEY: np.array([
                _WORKSPACE_SIZE / 2 + tolerance,
                _WORKSPACE_SIZE / 2 + tolerance,
                _WORKSPACE_SIZE / 2 + tolerance,
            ]),
        }),
        termination_checker.TerminationResult.TERMINATE,
    )


if __name__ == "__main__":
  absltest.main()
