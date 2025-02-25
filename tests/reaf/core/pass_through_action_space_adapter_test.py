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
"""Tests for the pass through action space adapter."""

from dm_env import specs
import numpy as np
from reaf.core import pass_through_action_space_adapter
from absl.testing import absltest

_COMMAND_KEY_1 = "command1"
_COMMAND_KEY_2 = "command2"
_COMMANDS_SPEC = {
    _COMMAND_KEY_1: specs.BoundedArray(
        shape=(5,), dtype=np.float32, minimum=1, maximum=10
    ),
    _COMMAND_KEY_2: specs.BoundedArray(
        shape=(5,), dtype=np.float32, minimum=1, maximum=10
    ),
}


class SequentialEnvironmentResetTest(absltest.TestCase):

  def test_commands_from_environment_action(self):
    adapter = pass_through_action_space_adapter.PassThroughActionSpaceAdapter(
        commands_spec=_COMMANDS_SPEC,
    )
    commands = {
        _COMMAND_KEY_1: np.array([1.0, 2.0, 3.0, 4.0, 5.0]).astype(np.float32),
        _COMMAND_KEY_2: np.array([6.0, 7.0, 8.0, 9.0, 10.0]).astype(np.float32),
    }
    self.assertEqual(
        adapter.commands_from_environment_action(environment_action=commands),
        commands,
    )

  def test_action_spec(self):
    adapter = pass_through_action_space_adapter.PassThroughActionSpaceAdapter(
        commands_spec=_COMMANDS_SPEC,
    )
    self.assertEqual(adapter.action_spec(), _COMMANDS_SPEC)

  def test_task_commands_keys(self):
    adapter = pass_through_action_space_adapter.PassThroughActionSpaceAdapter(
        commands_spec=_COMMANDS_SPEC,
    )
    self.assertSetEqual(
        adapter.task_commands_keys(), {_COMMAND_KEY_1, _COMMAND_KEY_2}
    )

  def test_passing_non_dict_action_raises_error(self):
    adapter = pass_through_action_space_adapter.PassThroughActionSpaceAdapter(
        commands_spec=_COMMANDS_SPEC,
    )
    with self.assertRaises(ValueError):
      adapter.commands_from_environment_action(
          environment_action=np.array([1.0, 2.0, 3.0, 4.0, 5.0])
      )

if __name__ == "__main__":
  absltest.main()
