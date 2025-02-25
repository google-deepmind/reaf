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
from collections.abc import Iterable

from dm_env import specs
import numpy as np
from reaf.common import partitioner_action_space_adapter

from absl.testing import absltest
from absl.testing import parameterized


class PartitionerActionSpaceAdapterTest(parameterized.TestCase):

  def test_action_spec(self):
    action_spec = specs.BoundedArray(
        shape=(5,),
        dtype=np.float32,
        minimum=-1.0,
        maximum=1.0,
    )
    adapter = partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
        partitions=(), action_spec=action_spec
    )
    self.assertEqual(action_spec, adapter.action_spec())

  def test_commands_keys(self):
    infos = (
        partitioner_action_space_adapter.PartitionInfo(
            command_key="command1",
        ),
        partitioner_action_space_adapter.PartitionInfo(
            command_key="command2",
            start_index=4,
        ),
    )
    adapter = partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
        partitions=infos,
        action_spec=specs.BoundedArray(
            shape=(9,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
        ),
    )
    self.assertEqual(
        adapter.task_commands_keys(),
        set((
            "command1",
            "command2",
        )),
    )

  def test_conversion(self):
    action = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float32)
    infos = (
        partitioner_action_space_adapter.PartitionInfo(
            command_key="command1",
            start_index=0,
            length=3,
        ),
        partitioner_action_space_adapter.PartitionInfo(
            command_key="command2",
            start_index=3,
        ),
        partitioner_action_space_adapter.PartitionInfo(
            command_key="command3",
            start_index=9,
            length=1,
        ),
    )
    adapter = partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
        partitions=infos,
        action_spec=specs.BoundedArray(
            shape=action.shape,
            dtype=np.float32,
            minimum=-1.0,
            maximum=10.0,
        ),
    )
    commands = adapter.commands_from_environment_action(action)

    np.testing.assert_equal(
        commands,
        {
            "command1": np.array([0, 1, 2]),
            "command2": np.array([3, 4, 5, 6, 7, 8, 9]),
            "command3": np.array([9]),
        },
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="single_partition_start_index_out_of_bounds",
          partitions=(
              partitioner_action_space_adapter.PartitionInfo(
                  command_key="a_key", start_index=6
              ),
          ),
      ),
      dict(
          testcase_name="multiple_partition_start_index_out_of_bounds",
          partitions=(
              partitioner_action_space_adapter.PartitionInfo(
                  command_key="a_key", start_index=0, length=6
              ),
              partitioner_action_space_adapter.PartitionInfo(
                  command_key="second_key", start_index=6
              ),
          ),
      ),
      dict(
          testcase_name="length_out_of_bounds",
          partitions=(
              partitioner_action_space_adapter.PartitionInfo(
                  command_key="a_key", start_index=0, length=7
              ),
          ),
      ),
  )
  def test_partition_out_of_bounds(
      self, partitions: Iterable[partitioner_action_space_adapter.PartitionInfo]
  ):
    with self.assertRaises(ValueError):
      partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
          partitions=partitions,
          action_spec=specs.BoundedArray(
              shape=(6,),
              dtype=np.float32,
              minimum=-1.0,
              maximum=10.0,
          ),
      )


if __name__ == "__main__":
  absltest.main()
