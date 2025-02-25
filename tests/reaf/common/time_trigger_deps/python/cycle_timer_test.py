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
# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for cycle_timer Python wrapper."""

import datetime

from reaf.common.time_trigger_deps.python import cycle_timer

from absl.testing import absltest


class CycleTimerTest(absltest.TestCase):

  def test_creation_with_default_clock(self):
    timer = cycle_timer.CycleTimer(datetime.timedelta(seconds=1))
    del timer

  def test_wait_next_period(self):
    timer = cycle_timer.CycleTimer(datetime.timedelta(seconds=0.5))

    before_call = datetime.datetime.now()
    # This should wait and return after 0.5s. We just care that this does not
    # get blocked.
    timer.wait_for_next_period()
    after_call = datetime.datetime.now()
    self.assertGreaterEqual(
        after_call - before_call, datetime.timedelta(seconds=0.5)
    )


if __name__ == "__main__":
  absltest.main()
