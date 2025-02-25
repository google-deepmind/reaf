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
"""Tests for sequential environment handlers."""

import dm_env
from reaf.common import sequential_end_of_episode_handler
from reaf.core import environment as reaf_environment
from absl.testing import absltest
from absl.testing import parameterized


class TestEnvironmentHandler(reaf_environment.EndOfEpisodeHandler):
  """Test environment handler."""

  def on_end_of_episode_stepping(self, timestep: dm_env.TimeStep) -> None:
    timestep.observation['count'] += 1


class SequentialEndOfEpisodeHandlerTest(parameterized.TestCase):

  @parameterized.parameters(0, 1, 2, 5, 10)
  def test_sequential_handler_calls_on_end_of_episode_stepping_for_all_handlers(
      self, count
  ):
    sequence_handler = (
        sequential_end_of_episode_handler.SequentialEndOfEpisodeHandler(
            [TestEnvironmentHandler() for _ in range(count)]
        )
    )
    timestep = dm_env.TimeStep(
        step_type=None, reward=None, discount=None, observation={'count': 0}
    )
    sequence_handler.on_end_of_episode_stepping(timestep)
    self.assertEqual(timestep.observation['count'], count)

    # Ensure you can run this multiple times without issue.
    sequence_handler.on_end_of_episode_stepping(timestep)
    self.assertEqual(timestep.observation['count'], count * 2)


if __name__ == '__main__':
  absltest.main()
