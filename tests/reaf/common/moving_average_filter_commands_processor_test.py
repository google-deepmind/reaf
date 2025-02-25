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
from dm_env import specs
import numpy as np
from reaf.common import moving_average_filter_commands_processor
from absl.testing import absltest
from absl.testing import parameterized


class MovingAverageFilterCommandsProcessorTest(parameterized.TestCase):

  def test_name(self):
    processor_name = 'test_processor'
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name=processor_name,
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1'],
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=-1, maximum=1
                )
            },
            average_window_size=3,
        ),
    )
    self.assertEqual(processor.name, processor_name)

  def test_process_commands(self):
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1'],
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=-1, maximum=1
                )
            },
            average_window_size=3,
        ),
    )

    # Initial commands (window not full yet)
    commands1 = {'command1': np.array([1, 0, -1], dtype=np.float32)}
    processed_commands1 = processor.process_commands(commands1)
    np.testing.assert_array_almost_equal(
        processed_commands1['command1'], np.array([1, 0, -1])
    )

    commands2 = {'command1': np.array([0.5, 0.2, -0.3], dtype=np.float32)}
    processed_commands2 = processor.process_commands(commands2)
    np.testing.assert_array_almost_equal(
        processed_commands2['command1'],
        np.array(
            [(1 + 0.5) / 2, (0 + 0.2) / 2, (-1 - 0.3) / 2], dtype=np.float32
        ),
    )

    commands3 = {'command1': np.array([0, 0.8, 0.2], dtype=np.float32)}
    processed_commands3 = processor.process_commands(commands3)
    np.testing.assert_array_almost_equal(
        processed_commands3['command1'],
        np.array(
            [(1 + 0.5 + 0) / 3, (0 + 0.2 + 0.8) / 3, (-1 - 0.3 + 0.2) / 3]
        ),
    )

    # Window is now full
    commands4 = {'command1': np.array([-0.2, -0.5, 0.9], dtype=np.float32)}
    processed_commands4 = processor.process_commands(commands4)
    np.testing.assert_array_almost_equal(
        processed_commands4['command1'],
        np.array(
            [(0.5 - 0.2 + 0) / 3, (0.2 + 0.8 - 0.5) / 3, (-0.3 + 0.2 + 0.9) / 3]
        ),
    )

    # Another command
    commands5 = {'command1': np.array([0.6, -0.1, -0.7], dtype=np.float32)}
    processed_commands5 = processor.process_commands(commands5)
    np.testing.assert_array_almost_equal(
        processed_commands5['command1'],
        np.array([
            (-0.2 + 0 + 0.6) / 3,
            (-0.5 + 0.8 - 0.1) / 3,
            (0.9 + 0.2 - 0.7) / 3,
        ]),
    )

  def test_process_commands_multiple_commands(self):
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1', 'command2'],
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=-1, maximum=1
                ),
                'command2': specs.BoundedArray(
                    shape=(2,), dtype=np.float64, minimum=0, maximum=100
                ),
            },
            average_window_size=2,
        ),
    )

    commands1 = {
        'command1': np.array([0.1, 0.2, 0.3], dtype=np.float32),
        'command2': np.array([10, 20], dtype=np.float64),
    }
    processed_commands1 = processor.process_commands(commands1)
    np.testing.assert_array_almost_equal(
        processed_commands1['command1'], np.array([0.1, 0.2, 0.3])
    )
    np.testing.assert_array_almost_equal(
        processed_commands1['command2'], np.array([10, 20])
    )

    commands2 = {
        'command1': np.array([0.4, 0.5, 0.6], dtype=np.float32),
        'command2': np.array([30, 40], dtype=np.float64),
    }
    processed_commands2 = processor.process_commands(commands2)
    np.testing.assert_array_almost_equal(
        processed_commands2['command1'],
        np.array(
            [(0.1 + 0.4) / 2, (0.2 + 0.5) / 2, (0.3 + 0.6) / 2],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_almost_equal(
        processed_commands2['command2'],
        np.array([(10 + 30) / 2, (20 + 40) / 2], dtype=np.float64),
    )

    # Check that they are independent
    commands3 = {
        'command1': np.array([0.7, 0.8, 0.9], dtype=np.float32),
        'command2': np.array([50, 60], dtype=np.float64),
    }
    processed_commands3 = processor.process_commands(commands3)
    np.testing.assert_array_almost_equal(
        processed_commands3['command1'],
        np.array([(0.4 + 0.7) / 2, (0.5 + 0.8) / 2, (0.6 + 0.9) / 2]),
    )
    np.testing.assert_array_almost_equal(
        processed_commands3['command2'],
        np.array([(30 + 50) / 2, (40 + 60) / 2], dtype=np.float64),
    )

  def test_reset(self):
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1'],
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=-2, maximum=2
                )
            },
            average_window_size=5,
        ),
    )

    # Add two commands to the window
    commands1 = {'command1': np.array([1, 1, 1], dtype=np.float32)}
    processor.process_commands(commands1)
    commands2 = {'command1': np.array([0, 0, 0], dtype=np.float32)}
    processor.process_commands(commands2)

    processor.reset()
    commands3 = {'command1': np.array([-1, -1, -1], dtype=np.float32)}
    processed_commands3 = processor.process_commands(
        commands3
    )  # Should be just command3, not average.
    np.testing.assert_array_almost_equal(
        processed_commands3['command1'],
        np.array([-1, -1, -1], dtype=np.float32),
    )

    processor.reset()
    # Check that validation flag has also been reset
    commands4 = {'command1': np.array([7, 8], dtype=np.float32)}  # Wrong shape
    with self.assertRaisesRegex(
        ValueError, "Shape mismatch for command 'command1'"
    ):
      processor.process_commands(commands4)

  def test_consumed_commands_spec(self):
    commands_spec = {
        'command1': specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=0, maximum=1
        ),
        'command2': specs.BoundedArray(
            shape=(2,), dtype=np.float64, minimum=-10, maximum=10
        ),
    }
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1', 'command2'],
            commands_spec=commands_spec,
            average_window_size=2,
        ),
    )
    self.assertEqual(processor.consumed_commands_spec(), commands_spec)

  def test_produced_commands_keys(self):
    commands_to_filter = ['command1', 'command2']
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=commands_to_filter,
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=0, maximum=1
                ),
                'command2': specs.BoundedArray(
                    shape=(2,), dtype=np.float64, minimum=-10, maximum=10
                ),
            },
            average_window_size=2,
        ),
    )
    self.assertEqual(
        processor.produced_commands_keys(), set(commands_to_filter)
    )

  def test_shape_mismatch_raises_value_error(self):
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1'],
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=-1, maximum=1
                )
            },
            average_window_size=3,
        ),
    )
    commands = {'command1': np.array([1, 2], dtype=np.float32)}  # Wrong shape
    with self.assertRaisesRegex(
        ValueError, "Shape mismatch for command 'command1'"
    ):
      processor.process_commands(commands)

  def test_dtype_mismatch_raises_value_error(self):
    processor = moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
        name='test',
        config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
            commands_to_filter=['command1'],
            commands_spec={
                'command1': specs.BoundedArray(
                    shape=(3,), dtype=np.float32, minimum=-1, maximum=1
                )
            },
            average_window_size=3,
        ),
    )
    commands = {'command1': np.array([1, 2, 3], dtype=np.int32)}  # Wrong dtype
    with self.assertRaisesRegex(
        ValueError, "Dtype mismatch for command 'command1'"
    ):
      processor.process_commands(commands)

  def test_mismatched_keys_raise_value_error(self):
    with self.assertRaisesRegex(
        ValueError,
        "The keys in 'commands_to_filter' and 'commands_spec' must match.",
    ):
      moving_average_filter_commands_processor.MovingAverageFilterCommandsProcessor(
          name='test',
          config=moving_average_filter_commands_processor.MovingAverageFilterConfig(
              commands_to_filter=['command1'],
              commands_spec={
                  'command2': specs.BoundedArray(
                      shape=(3,), dtype=np.float32, minimum=0, maximum=1
                  )
              },
              average_window_size=3,
          ),
      )


if __name__ == '__main__':
  absltest.main()
