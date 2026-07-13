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
"""Assertions for validating tree structures and specs."""

import unittest
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import tree


class TreeSpecValidationAssertions(unittest.TestCase):
  """Mix this into a googletest.TestCase class to get assertions for tree specs.

  Usage:

    class SomeTestCase(
      device_assertions.TreeSpecValidationAssertions,
      googletest.TestCase,
    ):
      ...
      def testSomething(self):
        ...
        self.assert_tree_matches_spec(tree_structure, spec)
  """

  def assert_tree_matches_spec(
      self,
      tree_structure: tree.Structure[gdmr_types.ArrayType],  # pyrefly: ignore[invalid-type-var]
      spec: tree.Structure[specs.Array],  # pyrefly: ignore[invalid-type-var]
  ):
    """Check that the tree_structure matches the spec.

    Args:
      tree_structure: the tree_structure that was calculated.
      spec: the spec for the tree_structure that should be validated.
    """
    if tree.is_nested(tree_structure):
      if len(tree_structure) != len(spec):  # pyrefly: ignore[bad-argument-type]
        self.fail(
            f"Spec mismatch. Spec: {spec} and the tree_structure:"
            f" {tree_structure}"
        )
      for idx, sub_tree in enumerate(tree_structure):  # pyrefly: ignore[bad-argument-type]
        self.assert_tree_matches_spec(sub_tree, spec[idx])  # pyrefly: ignore[bad-argument-type, bad-index]
    elif isinstance(spec, specs.Array):
      try:
        spec.validate(tree_structure)
      except Exception:  # pylint: disable=broad-exception-caught
        self.fail(
            f"Spec mismatch. Spec: {spec} and the tree_structure:"
            f" {tree_structure}"
        )
