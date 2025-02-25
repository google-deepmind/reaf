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
"""Termination checker based on an external signal."""

from collections.abc import Mapping
import threading

from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import termination_checker
from typing_extensions import override


class ExternalSignalTerminationChecker(termination_checker.TerminationChecker):
  """Checks if the episode should be terminated based on an external signal.

  This termination checker allows external users to control the termination of
  an episode. The user can call `do_terminate()` to signal that the episode
  should be terminated. Every subsequent call to `check_termination()` will
  return `TERMINATE` until the checker is reset by calling `reset()`, e.g. in an
  episode reset function.
  """

  def __init__(self, name: str):
    """Initializes the termination checker."""
    self._name = name

    self._mutex = threading.Lock()
    # Protected by _mutex
    self._checker_state = termination_checker.TerminationResult.DO_NOT_TERMINATE

  @override
  def reset(self) -> None:
    """Resets the termination checker."""
    with self._mutex:
      self._checker_state = (
          termination_checker.TerminationResult.DO_NOT_TERMINATE
      )

  def do_terminate(self) -> None:
    """Signal that the checker should return "TERMINATE" on the next call."""
    with self._mutex:
      self._checker_state = termination_checker.TerminationResult.TERMINATE

  def name(self) -> str:
    """Returns a unique string identifier for this object."""
    return self._name

  def check_termination(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> termination_checker.TerminationResult:
    """Checks if the episode should terminate."""
    del required_features
    with self._mutex:
      return self._checker_state

  def required_features_keys(self) -> set[str]:
    """Returns the feature keys that are required to check the termination."""
    return set()
