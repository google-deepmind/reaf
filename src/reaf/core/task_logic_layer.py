# Copyright 2026 Google LLC
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
"""DEPRECATED: Use reaf.core.task_layer instead.

This module is a backwards-compatible shim. The canonical class has been
renamed from TaskLogicLayer to TaskLayer and moved to reaf.core.task_layer.
"""

import warnings

from reaf.core.task_layer import TaskLayer


class TaskLogicLayer(TaskLayer):
  """Deprecated alias for TaskLayer.

  Use `from reaf.core.task_layer import TaskLayer` instead.
  """

  def __init__(self, *args, **kwargs):
    warnings.warn(
        "TaskLogicLayer was renamed to TaskLayer (semantics are unchanged). "
        "This class is deprecated, use TaskLayer instead. Import from "
        "reaf.core.task_layer.",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(*args, **kwargs)
