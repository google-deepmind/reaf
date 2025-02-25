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
"""Termination checker based on elapsed time."""

from collections.abc import Mapping
import time
from typing import Callable

from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import termination_checker
from typing_extensions import override


class MaximumWallClockTimeTerminationChecker(
    termination_checker.TerminationChecker
):
  """Terminate episode if it exceeds maximum elapsed wall-clock time."""

  def __init__(
      self,
      max_time_seconds: float | Callable[[], float],
      termination_result: termination_checker.TerminationResult = termination_checker.TerminationResult.TRUNCATE,
  ):
    """Constructor.

    Args:
      max_time_seconds: Maximum elapsed time in seconds (or a callable providing
        this value) after which a termination signal will be sent. The timer
        starts when `check_termination` is called for the first time after a
        reset.
      termination_result: The termination result when we reach the maximum time.
    """
    self._max_time_seconds: float = 0.0
    self._max_time_seconds_callable = None
    if isinstance(max_time_seconds, float):
      self._max_time_seconds = max_time_seconds
    elif isinstance(max_time_seconds, Callable):
      self._max_time_seconds_callable = max_time_seconds
      self._update_max_time_seconds()
    else:
      raise ValueError(
          'max_time_seconds must be a float or a callable that returns a float.'
      )
    self._start_time: float | None = None
    self._termination_result = termination_result

  def name(self) -> str:
    return 'maximum_wall_clock_time_termination_checker'

  def _update_max_time_seconds(self) -> float:
    if self._max_time_seconds_callable is not None:
      max_time_seconds = self._max_time_seconds_callable()
      if not isinstance(max_time_seconds, float):
        raise ValueError(
            'callable must return a float, but got'
            f' {type(max_time_seconds)}.'
        )
      self._max_time_seconds = max_time_seconds
    return self._max_time_seconds

  @override
  def reset(self):
    """Resets the timer start time."""
    if self._max_time_seconds_callable is not None:
      self._max_time_seconds = self._max_time_seconds_callable()
    self._start_time = None

  @override
  def check_termination(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> termination_checker.TerminationResult:
    """Check episode termination if the episode exceeds max elapsed time."""
    if self._start_time is None:
      self._start_time = time.monotonic()

    elapsed_time = time.monotonic() - self._start_time

    if elapsed_time >= self._max_time_seconds:
      return self._termination_result
    else:
      return termination_checker.TerminationResult.DO_NOT_TERMINATE

  @override
  def required_features_keys(self) -> set[str]:
    """Returns the keys of the required features."""
    return set([])
