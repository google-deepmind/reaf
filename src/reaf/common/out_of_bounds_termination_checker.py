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
"""Out of bounds termination checker."""

from collections.abc import Mapping

from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import termination_checker


class OutOfBoundsTerminationChecker(termination_checker.TerminationChecker):
  """Checks if a set of features are out of bounds."""

  def __init__(
      self,
      name: str,
      feature_bounds: Mapping[
          str, tuple[gdmr_types.ArrayType, gdmr_types.ArrayType]
      ],
      termination_result: termination_checker.TerminationResult = (
          termination_checker.TerminationResult.TERMINATE
      ),
  ):
    """Initializes the checker.

    Args:
      name: The name of the checker.
      feature_bounds: A dictionary mapping feature names to a tuple of
        (min_values, max_values), where min_values and max_values are arrays of
        the same shape as the corresponding feature.
      termination_result: The type of termination to return when a feature is
        out of bounds.
    """
    self._name = name
    self._feature_bounds = feature_bounds
    self._termination_result = termination_result

  def name(self) -> str:
    """Returns the name of the checker."""
    return self._name

  def check_termination(
      self, required_features: Mapping[str, np.ndarray]
  ) -> termination_checker.TerminationResult:
    """Checks if any feature is out of bounds."""
    for feature_name, (min_values, max_values) in self._feature_bounds.items():
      feature_values = required_features[feature_name]
      if np.any(
          np.logical_or(
              feature_values <= min_values, feature_values >= max_values
          )
      ):
        return self._termination_result
    return termination_checker.TerminationResult.DO_NOT_TERMINATE

  def required_features_keys(self) -> set[str]:
    """The keys of the features that are required by this checker."""
    return set(self._feature_bounds.keys())
