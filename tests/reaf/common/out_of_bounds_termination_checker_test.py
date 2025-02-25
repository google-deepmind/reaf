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
"""Tests for out of bounds termination checker."""

import numpy as np
from reaf.common import out_of_bounds_termination_checker
from reaf.core import termination_checker
from absl.testing import absltest


class OutOfBoundsTerminationCheckerTest(absltest.TestCase):

  def test_check_termination_name(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
    )
    self.assertEqual('test', checker.name())

  def test_check_termination_in_bounds(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
    )
    required_features = {
        'feature1': np.array([0.5, 1.5]),
        'feature2': np.array([0.0]),
    }
    self.assertEqual(
        termination_checker.TerminationResult.DO_NOT_TERMINATE,
        checker.check_termination(required_features),
    )

  def test_check_termination_out_of_bounds_min(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
    )
    required_features = {
        'feature1': np.array([-0.5, 1.5]),
        'feature2': np.array([0.0]),
    }
    self.assertEqual(
        termination_checker.TerminationResult.TERMINATE,
        checker.check_termination(required_features),
    )

  def test_check_termination_out_of_bounds_max(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
    )
    required_features = {
        'feature1': np.array([0.5, 2.5]),
        'feature2': np.array([0.0]),
    }
    self.assertEqual(
        termination_checker.TerminationResult.TERMINATE,
        checker.check_termination(required_features),
    )

  def test_check_termination_multiple_out_of_bounds(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
    )
    required_features = {
        'feature1': np.array([-0.5, 2.5]),
        'feature2': np.array([0.0]),
    }
    self.assertEqual(
        termination_checker.TerminationResult.TERMINATE,
        checker.check_termination(required_features),
    )

  def test_check_termination_truncate(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
        termination_result=termination_checker.TerminationResult.TRUNCATE,
    )
    required_features = {
        'feature1': np.array([-0.5, 2.5]),
        'feature2': np.array([0.0]),
    }
    self.assertEqual(
        termination_checker.TerminationResult.TRUNCATE,
        checker.check_termination(required_features),
    )

  def test_required_features_keys(self):
    checker = out_of_bounds_termination_checker.OutOfBoundsTerminationChecker(
        name='test',
        feature_bounds={
            'feature1': (np.array([0.0, 1.0]), np.array([1.0, 2.0])),
            'feature2': (np.array([-1.0]), np.array([1.0])),
        },
    )
    self.assertEqual({'feature1', 'feature2'}, checker.required_features_keys())


if __name__ == '__main__':
  absltest.main()
