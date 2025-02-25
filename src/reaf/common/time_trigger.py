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
"""REAF trigger based on a fixed time interval."""

import datetime
from reaf.common.time_trigger_deps.python import cycle_timer
from reaf.core import trigger


class TimeTrigger(trigger.Trigger):
  """Trigger that fires at a fixed time interval."""

  def __init__(self, period: datetime.timedelta, name: str = "TimeTrigger"):
    if period.total_seconds() <= 0:
      raise ValueError(
          f"Period must be positive. Got {period.total_seconds()}."
      )
    self._timer = cycle_timer.CycleTimer(period)
    self._name = name

  @property
  def name(self) -> str:
    return self._name

  def wait_for_event(self) -> None:
    self._timer.wait_for_next_period()
