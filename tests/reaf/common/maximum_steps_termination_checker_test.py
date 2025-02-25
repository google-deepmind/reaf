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
"""Tests for step based termination checker."""

from reaf.common import maximum_steps_termination_checker
from reaf.core import termination_checker
from absl.testing import absltest
from absl.testing import parameterized


class MockStepsGetter:
  """Mock class for getting the number of steps."""

  def __init__(self, steps: int):
    self._steps = steps

  def get_steps(self) -> int:
    return self._steps

  def set_steps(self, steps: int):
    self._steps = steps


class MaximumStepsTerminationCheckerTest(parameterized.TestCase):

  def test_reset_resets_counter(self):
    max_steps = 2
    checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
        max_steps
    )
    # Take 2 steps
    checker.check_termination({})
    termination_result = checker.check_termination({})
    self.assertTrue(termination_result.is_truncated())
    checker.reset()
    # Take 1 step
    termination_result = checker.check_termination({})
    self.assertFalse(termination_result.is_truncated())

  def test_terminates_if_called_more_than_or_equal_to_max_steps(self):
    max_steps = 4
    checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
        max_steps
    )
    self.assertEmpty(checker.required_features_keys())
    for step in range(max_steps):
      termination_result = checker.check_termination({})
      if step == max_steps - 1:
        self.assertTrue(termination_result.is_truncated())
      else:
        self.assertFalse(termination_result.is_truncated())

  def test_required_features_keys(self):
    max_steps = 2
    checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
        max_steps
    )
    self.assertEmpty(checker.required_features_keys())

  @parameterized.parameters(
      termination_checker.TerminationResult.TRUNCATE,
      termination_checker.TerminationResult.TERMINATE,
  )
  def test_termination_result(self, termination_result):
    max_steps = 10
    checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
        max_steps, termination_result
    )

    for _ in range(max_steps - 1):
      self.assertEqual(
          checker.check_termination({}),
          termination_checker.TerminationResult.DO_NOT_TERMINATE,
      )

    self.assertEqual(checker.check_termination({}), termination_result)

  def test_set_max_steps(self):
    max_steps = 4
    checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
        max_steps
    )
    new_max_steps = max_steps * 2
    checker.set_max_steps(new_max_steps)
    for step in range(new_max_steps):
      termination_result = checker.check_termination({})
      if step == new_max_steps - 1:
        self.assertTrue(termination_result.is_truncated())
      else:
        self.assertFalse(termination_result.is_truncated())

  def test_set_max_steps_callable(self):
    steps_getter = MockStepsGetter(4)
    checker = maximum_steps_termination_checker.MaximumStepsTerminationChecker(
        steps_getter.get_steps,
    )
    new_steps = 8
    steps_getter.set_steps(new_steps)
    checker.reset()
    for step in range(new_steps):
      termination_result = checker.check_termination({})
      if step == new_steps - 1:
        self.assertTrue(termination_result.is_truncated())
      else:
        self.assertFalse(termination_result.is_truncated())

if __name__ == '__main__':
  absltest.main()
