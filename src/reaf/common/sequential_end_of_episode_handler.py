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
"""Sequential end of episode handler for agent."""

from collections.abc import Sequence

import dm_env
from reaf.core import environment as reaf_environment


class SequentialEndOfEpisodeHandler(reaf_environment.EndOfEpisodeHandler):
  """Sequentially call different end of episode handlers."""

  def __init__(
      self,
      end_of_episode_handlers: Sequence[reaf_environment.EndOfEpisodeHandler],
  ):
    super().__init__()
    self._handlers = end_of_episode_handlers

  def on_end_of_episode_stepping(
      self,
      timestep: dm_env.TimeStep,
  ) -> None:
    """Runs the end of episode handlers in sequence."""
    for handler in self._handlers:
      handler.on_end_of_episode_stepping(timestep)
