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
import time
from unittest import mock

from reaf.common import maximum_wall_clock_time_termination_checker
from reaf.core import termination_checker

from absl.testing import absltest
from absl.testing import parameterized


class MockTimeGetter:
  """Mock class for getting the current time."""

  def __init__(self, time_seconds: float):
    self._time_seconds = time_seconds

  def get_time_seconds(self) -> float:
    return self._time_seconds

  def set_time_seconds(self, time_seconds: float):
    self._time_seconds = time_seconds


class MaximumWallClockTimeTerminationCheckerTest(parameterized.TestCase):

  @mock.patch.object(time, 'monotonic')
  def test_timer_starts_on_first_check(self, mock_monotonic):
    max_time_seconds = 5.0
    checker = maximum_wall_clock_time_termination_checker.MaximumWallClockTimeTerminationChecker(
        max_time_seconds
    )
    mock_monotonic.return_value = 100.0

    # Before first check, start_time should be None
    self.assertIsNone(checker._start_time)  # pylint: disable=protected-access

    # First check should start the timer
    checker.check_termination({})
    self.assertEqual(checker._start_time, 100.0)  # pylint: disable=protected-access

    # Subsequent check should not reset the timer
    mock_monotonic.return_value = 101.0
    checker.check_termination({})
    self.assertEqual(checker._start_time, 100.0)  # pylint: disable=protected-access

  @mock.patch.object(time, 'monotonic')
  def test_terminates_after_max_time(self, mock_monotonic):
    max_time_seconds = 5.0
    checker = maximum_wall_clock_time_termination_checker.MaximumWallClockTimeTerminationChecker(
        max_time_seconds
    )

    # Initial time
    mock_monotonic.return_value = 100.0
    termination_result = checker.check_termination({})
    self.assertEqual(
        termination_result,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

    # Time before limit
    mock_monotonic.return_value = 100.0 + max_time_seconds - 0.1
    termination_result = checker.check_termination({})
    self.assertEqual(
        termination_result,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

    # Time at limit
    mock_monotonic.return_value = 100.0 + max_time_seconds
    termination_result = checker.check_termination({})
    self.assertTrue(termination_result.is_truncated())

    # Time after limit
    mock_monotonic.return_value = 100.0 + max_time_seconds + 0.1
    termination_result = checker.check_termination({})
    self.assertTrue(termination_result.is_truncated())

  @mock.patch.object(time, 'monotonic')
  def test_reset_resets_timer(self, mock_monotonic):
    max_time_seconds = 5.0
    checker = maximum_wall_clock_time_termination_checker.MaximumWallClockTimeTerminationChecker(
        max_time_seconds
    )

    # Run until termination
    mock_monotonic.return_value = 100.0
    checker.check_termination({})
    mock_monotonic.return_value = 100.0 + max_time_seconds
    termination_result = checker.check_termination({})
    self.assertTrue(termination_result.is_truncated())

    # Reset and check again
    checker.reset()
    self.assertIsNone(checker._start_time)  # pylint: disable=protected-access
    mock_monotonic.return_value = 110.0  # New start time
    termination_result = checker.check_termination({})
    self.assertEqual(
        termination_result,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )
    self.assertEqual(checker._start_time, 110.0)  # pylint: disable=protected-access

  def test_required_features_keys(self):
    checker = maximum_wall_clock_time_termination_checker.MaximumWallClockTimeTerminationChecker(
        max_time_seconds=1.0
    )
    self.assertEmpty(checker.required_features_keys())

  @parameterized.parameters(
      termination_checker.TerminationResult.TRUNCATE,
      termination_checker.TerminationResult.TERMINATE,
  )
  @mock.patch.object(time, 'monotonic')
  def test_termination_result(
      self, expected_termination_result, mock_monotonic
  ):
    max_time_seconds = 10.0
    checker = maximum_wall_clock_time_termination_checker.MaximumWallClockTimeTerminationChecker(
        max_time_seconds, expected_termination_result
    )

    mock_monotonic.return_value = 100.0
    self.assertEqual(
        checker.check_termination({}),
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

    mock_monotonic.return_value = 100.0 + max_time_seconds
    self.assertEqual(checker.check_termination({}), expected_termination_result)

  @mock.patch.object(time, 'monotonic')
  def test_terminates_after_max_time_with_time_getter(self, mock_monotonic):
    time_getter = MockTimeGetter(5.0)
    checker = maximum_wall_clock_time_termination_checker.MaximumWallClockTimeTerminationChecker(
        time_getter.get_time_seconds
    )
    time_getter.set_time_seconds(10.0)
    checker.reset()

    # Initial time
    mock_monotonic.return_value = 100.0
    termination_result = checker.check_termination({})
    self.assertEqual(
        termination_result,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

    # Time before limit
    mock_monotonic.return_value = 100.0 + time_getter.get_time_seconds() - 0.1
    termination_result = checker.check_termination({})
    self.assertEqual(
        termination_result,
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

    # Time at limit
    mock_monotonic.return_value = 100.0 + time_getter.get_time_seconds()
    termination_result = checker.check_termination({})
    self.assertTrue(termination_result.is_truncated())

    # Time after limit
    mock_monotonic.return_value = 100.0 + time_getter.get_time_seconds() + 0.1
    termination_result = checker.check_termination({})
    self.assertTrue(termination_result.is_truncated())


if __name__ == '__main__':
  absltest.main()
