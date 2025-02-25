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
"""Termination checker for agent timestep."""

from collections.abc import Mapping
from typing import Callable

from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import termination_checker
from typing_extensions import override


class MaximumStepsTerminationChecker(termination_checker.TerminationChecker):
  """Terminate episode if it exceeds maximum number of steps."""

  def __init__(
      self,
      max_steps: int | Callable[[], int],
      termination_result: termination_checker.TerminationResult = termination_checker.TerminationResult.TRUNCATE,
  ):
    """Constructor.

    Args:
      max_steps: Maximum number of environment steps after which a termination
        signal will be sent or a callable that provides this value.
      termination_result: The termination result when we reach the maximum
        number of steps.
    """
    self._max_steps_callable: Callable[[], int] | None = None
    self._max_steps: int = 0
    if isinstance(max_steps, Callable):
      self._max_steps_callable = max_steps
    elif isinstance(max_steps, int):
      self._max_steps = max_steps
    else:
      raise ValueError(
          'max_steps must be an int or a callable that returns an int.'
      )
    self._current_number_of_steps: int = 0
    self._termination_result = termination_result

  def name(self) -> str:
    return 'maximum_steps_termination_checker'

  def _get_max_steps(self) -> int:
    # Do not put _update_max_steps in this function because this is called every
    # step and calling the callable can sometimes be expensive if contacting
    # external services.
    return self._max_steps

  def _update_max_steps(self) -> None:
    if self._max_steps_callable is not None:
      if not isinstance(self._max_steps_callable(), int):
        raise ValueError(
            'callable must return an int, but got'
            f' {type(self._max_steps_callable())}.'
        )
      self._max_steps = self._max_steps_callable()

  @override
  def reset(self):
    self._update_max_steps()
    self._current_number_of_steps = 0

  @override
  def check_termination(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> termination_checker.TerminationResult:
    """Check episode termination if the episode exceeds max number of steps."""
    # This method gets called only once per step within REAF Task Layer.
    self._current_number_of_steps += 1
    if self._current_number_of_steps >= self._get_max_steps():
      return self._termination_result
    else:
      return termination_checker.TerminationResult.DO_NOT_TERMINATE

  @override
  def required_features_keys(self) -> set[str]:
    """Returns the keys of the required features."""
    return set([])

  def set_max_steps(self, max_steps: int) -> None:
    self._max_steps = max_steps
