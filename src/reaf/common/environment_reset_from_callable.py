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
"""Reset environment from a callable."""

from collections.abc import Callable
from gdm_robotics.interfaces import environment as gdmr_env
from reaf.core import environment as reaf_environment


class EnvironmentResetFromCallable(reaf_environment.EnvironmentReset):
  """Creates an EnvironmentReset from a callable."""

  def __init__(self, reset_callable: Callable[[gdmr_env.ResetOptions], None]):
    self._callable = reset_callable

  def do_reset(self, reset_options: gdmr_env.ResetOptions):
    self._callable(reset_options)
