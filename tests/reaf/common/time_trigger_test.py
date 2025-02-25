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
"""Tests for the time_trigger Python module."""

import datetime
from reaf.common import time_trigger
from absl.testing import absltest


class TimeTriggerTest(absltest.TestCase):

  def test_negative_period_raises_error(self):
    with self.assertRaises(ValueError):
      time_trigger.TimeTrigger(period=datetime.timedelta(seconds=-1))

  def test_wait_next_period(self):
    trigger = time_trigger.TimeTrigger(
        period=datetime.timedelta(milliseconds=500)
    )
    before_time = datetime.datetime.now()
    trigger.wait_for_event()
    after_time = datetime.datetime.now()
    self.assertGreaterEqual(
        after_time - before_time, datetime.timedelta(milliseconds=500)
    )
    # We do not test the interval to be less than or exact the period as there
    # are no guarantees that the scheduler will be able to run the trigger
    # exactly at the right time, especially on Forge.


if __name__ == "__main__":
  absltest.main()
