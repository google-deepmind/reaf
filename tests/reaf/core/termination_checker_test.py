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
from collections.abc import Sequence

from reaf.core import termination_checker

from absl.testing import absltest
from absl.testing import parameterized


class TerminationResultTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="terminated",
          termination_result=termination_checker.TerminationResult.TERMINATE,
          expected_result=True,
      ),
      dict(
          testcase_name="not_terminated",
          termination_result=termination_checker.TerminationResult.DO_NOT_TERMINATE,
          expected_result=False,
      ),
      dict(
          testcase_name="truncated",
          termination_result=termination_checker.TerminationResult.TRUNCATE,
          expected_result=False,
      ),
  )
  def test_is_terminated(
      self,
      termination_result: termination_checker.TerminationResult,
      expected_result: bool,
  ):
    self.assertEqual(expected_result, termination_result.is_terminated())

  @parameterized.named_parameters(
      dict(
          testcase_name="terminated",
          termination_result=termination_checker.TerminationResult.TERMINATE,
          expected_result=False,
      ),
      dict(
          testcase_name="not_terminated",
          termination_result=termination_checker.TerminationResult.DO_NOT_TERMINATE,
          expected_result=False,
      ),
      dict(
          testcase_name="truncated",
          termination_result=termination_checker.TerminationResult.TRUNCATE,
          expected_result=True,
      ),
  )
  def test_is_truncated(
      self,
      termination_result: termination_checker.TerminationResult,
      expected_result: bool,
  ):
    self.assertEqual(expected_result, termination_result.is_truncated())

  @parameterized.named_parameters(
      dict(
          testcase_name="not_terminated",
          termination_results=(
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
          ),
          expected_combined_result=termination_checker.TerminationResult.DO_NOT_TERMINATE,
      ),
      dict(
          testcase_name="at_least_a_termination_terminates",
          termination_results=(
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.TRUNCATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.TRUNCATE,
              termination_checker.TerminationResult.TERMINATE,
              termination_checker.TerminationResult.TRUNCATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
          ),
          expected_combined_result=termination_checker.TerminationResult.TERMINATE,
      ),
      dict(
          testcase_name="one_truncation_truncates",
          termination_results=(
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
              termination_checker.TerminationResult.TRUNCATE,
              termination_checker.TerminationResult.DO_NOT_TERMINATE,
          ),
          expected_combined_result=termination_checker.TerminationResult.TRUNCATE,
      ),
      dict(
          testcase_name="termination_wins_on_truncation",
          termination_results=(
              termination_checker.TerminationResult.TRUNCATE,
              termination_checker.TerminationResult.TRUNCATE,
              termination_checker.TerminationResult.TERMINATE,
          ),
          expected_combined_result=termination_checker.TerminationResult.TERMINATE,
      ),
  )
  def test_result_combination(
      self,
      termination_results: Sequence[termination_checker.TerminationResult],
      expected_combined_result: termination_checker.TerminationResult,
  ):
    # The test has at least one element in the sequence.
    current_state = termination_results[0]
    for termination_result in termination_results[1:]:
      current_state = termination_result.combine(current_state)

    self.assertEqual(expected_combined_result, current_state)


if __name__ == "__main__":
  absltest.main()
