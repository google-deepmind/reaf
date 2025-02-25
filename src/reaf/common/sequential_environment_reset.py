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
"""Sequential environment reset for agent."""

from collections.abc import Iterable
from gdm_robotics.interfaces import environment as gdmr_env
from reaf.core import environment as reaf_environment


class SequentialEnvironmentReset(reaf_environment.EnvironmentReset):
  """Sequentially call different environment resetters."""

  def __init__(self, resets: Iterable[reaf_environment.EnvironmentReset]):
    super().__init__()
    self._resets = list(resets)

  def do_reset(
      self,
      config: gdmr_env.ResetOptions = gdmr_env.Options(),
  ) -> None:
    """Resets the environment."""
    for reset in self._resets:
      reset.do_reset(config)
