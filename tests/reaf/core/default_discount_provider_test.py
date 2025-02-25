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
from dm_env import specs
import numpy as np
from reaf.core import default_discount_provider
from reaf.core import termination_checker
from absl.testing import absltest
from absl.testing import parameterized


class DefaultDiscountProviderTest(parameterized.TestCase):

  def test_spec_is_correct(self):
    self.assertEqual(
        specs.BoundedArray(
            shape=(),
            dtype=np.float64,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        ),
        default_discount_provider.DefaultDiscountProvider().discount_spec(),
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="not_terminate",
          termination_state=termination_checker.TerminationResult.DO_NOT_TERMINATE,
          expected_discount=np.array([1.0]),
      ),
      dict(
          testcase_name="terminate",
          termination_state=termination_checker.TerminationResult.TERMINATE,
          expected_discount=np.array([0.0]),
      ),
      dict(
          testcase_name="truncate",
          termination_state=termination_checker.TerminationResult.TRUNCATE,
          expected_discount=np.array([1.0]),
      ),
  )
  def test_discount_is_correct(
      self,
      termination_state: termination_checker.TerminationResult,
      expected_discount: np.ndarray,
  ):
    np.testing.assert_equal(
        default_discount_provider.DefaultDiscountProvider().compute_discount(
            {}, termination_state
        ),
        expected_discount,
    )


if __name__ == "__main__":
  absltest.main()
