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
from reaf.common import external_signal_termination_checker
from reaf.core import termination_checker
from absl.testing import absltest


class ExternalSignalTerminationCheckerTest(absltest.TestCase):

  def test_name_property(self):
    checker = (
        external_signal_termination_checker.ExternalSignalTerminationChecker(
            name="test"
        )
    )
    self.assertEqual(checker.name(), "test")

  def test_required_features_keys_is_empty(self):
    checker = (
        external_signal_termination_checker.ExternalSignalTerminationChecker(
            name="test"
        )
    )
    self.assertEmpty(checker.required_features_keys())

  def test_no_signal_does_not_terminate(self):
    checker = (
        external_signal_termination_checker.ExternalSignalTerminationChecker(
            name="test"
        )
    )
    self.assertEqual(
        checker.check_termination(required_features={}),
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )

  def test_signal_terminates(self):
    checker = (
        external_signal_termination_checker.ExternalSignalTerminationChecker(
            name="test"
        )
    )
    checker.do_terminate()
    self.assertEqual(
        checker.check_termination(required_features={}),
        termination_checker.TerminationResult.TERMINATE,
    )

  def test_reset_clears_signal(self):
    checker = (
        external_signal_termination_checker.ExternalSignalTerminationChecker(
            name="test"
        )
    )
    checker.do_terminate()
    self.assertEqual(
        checker.check_termination(required_features={}),
        termination_checker.TerminationResult.TERMINATE,
    )
    checker.reset()
    self.assertEqual(
        checker.check_termination(required_features={}),
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
    )


if __name__ == "__main__":
  absltest.main()
