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
"""Test TestRewardProvider module."""
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.testing import tree_spec_validation_assertions
import tree
from absl.testing import absltest
from absl.testing import parameterized

_NP_FLOAT_ARRAY = np.array([1.0, 2.0, 3.0], dtype=float)
_ANOTHER_NP_FLOAT_ARRAY = np.array([4.0, 5.0], dtype=float)
_NESTED_NP_FLOAT_ARRAY = np.array(
    [_NP_FLOAT_ARRAY, _ANOTHER_NP_FLOAT_ARRAY], dtype=object
)


class TreeSpecValidationAssertionsTest(
    tree_spec_validation_assertions.TreeSpecValidationAssertions,
    parameterized.TestCase,
):

  @parameterized.named_parameters(
      dict(
          testcase_name="not_nested",
          tree_structure=_NP_FLOAT_ARRAY,
          spec=specs.Array(
              shape=_NP_FLOAT_ARRAY.shape, dtype=_NP_FLOAT_ARRAY.dtype
          ),
      ),
      dict(
          testcase_name="nested",
          tree_structure=_NESTED_NP_FLOAT_ARRAY,
          spec=specs.Array(
              shape=_NESTED_NP_FLOAT_ARRAY.shape,
              dtype=_NESTED_NP_FLOAT_ARRAY.dtype,
          ),
      ),
  )
  def test_assert_tree_matches_spec_success(
      self,
      tree_structure: tree.Structure[gdmr_types.ArrayType],
      spec: tree.Structure[specs.Array],
  ):
    self.assert_tree_matches_spec(tree_structure, spec)

  @parameterized.named_parameters(
      dict(
          testcase_name="wrong_shape",
          tree_structure=_NP_FLOAT_ARRAY,
          spec=specs.Array(shape=(2, 1), dtype=_NP_FLOAT_ARRAY.dtype),
      ),
      dict(
          testcase_name="wrong_dtype",
          tree_structure=_NESTED_NP_FLOAT_ARRAY,
          spec=specs.Array(shape=_NESTED_NP_FLOAT_ARRAY.shape, dtype=int),
      ),
  )

  @absltest.skip("expectedFailure fails with pytest")
  def test_assert_tree_matches_spec_failure(
      self,
      tree_structure: tree.Structure[gdmr_types.ArrayType],
      spec: tree.Structure[specs.Array],
  ):
    self.assert_tree_matches_spec(tree_structure, spec)


if __name__ == "__main__":
  absltest.main()
