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
"""Tests for resetting environment from callable."""

from gdm_robotics.interfaces import environment as gdmr_env
from reaf.common import environment_reset_from_callable
from absl.testing import absltest


class EnvironmentResetFromCallableTest(absltest.TestCase):

  def test_reset_callable_is_invoked(self):
    step_count = 10
    def reset_step_count_fn(options: gdmr_env.ResetOptions):
      del options
      nonlocal step_count
      step_count = 0
    env_resetter = environment_reset_from_callable.EnvironmentResetFromCallable(
        reset_step_count_fn
    )
    env_resetter.do_reset(gdmr_env.Options())
    self.assertEqual(step_count, 0)


if __name__ == "__main__":
  absltest.main()
