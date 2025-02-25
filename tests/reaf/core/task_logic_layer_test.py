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
from collections.abc import Mapping, Sequence
from unittest import mock

from dm_env import specs
import numpy as np
from reaf.core import commands_processor as reaf_commands_processor
from reaf.core import discount_provider as reaf_discount_provider
from reaf.core import features_observer as reaf_features_observers
from reaf.core import features_producer as reaf_features_producer
from reaf.core import logger as reaf_logger
from reaf.core import numpy_mock_assertions
from reaf.core import reward_provider as reaf_reward_provider
from reaf.core import task_logic_layer
from reaf.core import termination_checker as reaf_termination_checker

from absl.testing import absltest
from absl.testing import parameterized


def _commands_processor(
    spec: Mapping[str, specs.Array], keys: set[str]
) -> reaf_commands_processor.CommandsProcessor:
  processor = mock.create_autospec(
      reaf_commands_processor.CommandsProcessor, instance=True
  )
  processor.consumed_commands_spec.return_value = spec
  processor.produced_commands_keys.return_value = keys
  return processor


def _features_producer(
    spec: Mapping[str, specs.Array], required_keys: set[str]
) -> reaf_features_producer.FeaturesProducer:
  producer = mock.create_autospec(
      reaf_features_producer.FeaturesProducer, instance=True
  )
  producer.produced_features_spec.return_value = spec
  producer.required_features_keys.return_value = required_keys
  return producer


class TaskLogicLayerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="no_processing",
          commands_processors=(),
          features_producers=(),
      ),
      dict(
          testcase_name="with_processing",
          commands_processors=(
              _commands_processor(
                  {}, set(("processor1/command1", "dacl/command3"))
              ),
              _commands_processor(
                  {
                      "processor1/command1": specs.Array(
                          shape=(1,), dtype=np.float32
                      )
                  },
                  set(("dacl/command1", "dacl/command2")),
              ),
          ),
          features_producers=(
              _features_producer(
                  {"prod1/feature1": specs.Array(shape=(3,), dtype=np.int32)},
                  set(("dacl/measurement1",)),
              ),
              _features_producer(
                  {"prod2/feature1": specs.Array(shape=(1,), dtype=np.float32)},
                  set(("dacl/measurement1", "prod1/feature1")),
              ),
          ),
      ),
  )
  def test_valid_command_and_features_specs(
      self,
      commands_processors: Sequence[reaf_commands_processor.CommandsProcessor],
      features_producers: Sequence[reaf_features_producer.FeaturesProducer],
  ):
    dacl_commands_spec = {
        "dacl/command1": specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
        "dacl/command2": specs.Array(shape=(3,), dtype=np.float16),
        "dacl/command3": specs.StringArray(shape=()),
    }
    dacl_measurements_spec = {
        "dacl/measurement1": specs.BoundedArray(
            shape=(5,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
        "dacl/measurement2": specs.Array(shape=(1,), dtype=np.int32),
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=commands_processors,
        features_producers=features_producers,
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    task_layer.validate_spec(
        dacl_commands_spec=dacl_commands_spec,
        dacl_measurements_spec=dacl_measurements_spec,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="commands_processing_do_not_match_dacl_spec",
          commands_processors=(
              _commands_processor({}, set(("dacl/missing_command",))),
          ),
          features_producers=(),
      ),
      dict(
          testcase_name="commands_processing_do_not_match_next_processor",
          commands_processors=(
              _commands_processor(
                  {}, set(("processor1/foo_command", "dacl/command3"))
              ),
              _commands_processor(
                  {
                      "processor1/bar_command": specs.Array(
                          shape=(1,), dtype=np.float32
                      )
                  },
                  set(("dacl/command1", "dacl/command2")),
              ),
          ),
          features_producers=(),
      ),
      dict(
          testcase_name="features_producer_uses_missing_dacl_measurment",
          commands_processors=(),
          features_producers=(
              _features_producer(
                  {
                      "producer1/feature1": specs.Array(
                          shape=(3,), dtype=np.int32
                      )
                  },
                  set(("dacl/measurement3",)),
              ),
          ),
      ),
      dict(
          testcase_name="features_producer_wrong_pipeline",
          commands_processors=(),
          features_producers=(
              _features_producer(
                  {
                      "producer1/foo_feature": specs.Array(
                          shape=(3,), dtype=np.int32
                      )
                  },
                  set(("dacl/measurement1",)),
              ),
              _features_producer(
                  {
                      "producer2/feature1": specs.Array(
                          shape=(1,), dtype=np.float32
                      )
                  },
                  set(("dacl/measurement1", "producer1/bar_feature")),
              ),
          ),
      ),
      dict(
          testcase_name="duplicate_key_in_features_producer",
          commands_processors=(),
          features_producers=(
              _features_producer(
                  {
                      "producer1/feature1": specs.Array(
                          shape=(3,), dtype=np.int32
                      )
                  },
                  set(("dacl/measurement1",)),
              ),
              _features_producer(
                  {
                      "producer1/feature1": specs.Array(
                          shape=(1,), dtype=np.float16
                      ),
                  },
                  set(("dacl/measurement1", "producer1/feature1")),
              ),
          ),
      ),
      dict(
          testcase_name="command_processor_removes_required_dacl_spec",
          commands_processors=(
              _commands_processor(
                  {}, set(("processor1/command1", "dacl/command3"))
              ),
              _commands_processor(
                  {"dacl/command1": specs.Array(shape=(1,), dtype=np.float32)},
                  set(("dacl/command2",)),
              ),
          ),
          features_producers=(),
      ),
  )
  def test_wrong_processing_spec_raises_error(
      self,
      commands_processors: Sequence[reaf_commands_processor.CommandsProcessor],
      features_producers: Sequence[reaf_features_producer.FeaturesProducer],
  ):
    dacl_commands_spec = {
        "dacl/command1": specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
        "dacl/command2": specs.Array(shape=(3,), dtype=np.float16),
        "dacl/command3": specs.StringArray(shape=()),
    }
    dacl_measurements_spec = {
        "dacl/measurement1": specs.BoundedArray(
            shape=(5,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
        "dacl/measurement2": specs.Array(shape=(1,), dtype=np.int32),
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=commands_processors,
        features_producers=features_producers,
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    with self.assertRaises(ValueError):
      task_layer.validate_spec(
          dacl_commands_spec=dacl_commands_spec,
          dacl_measurements_spec=dacl_measurements_spec,
      )

  def test_features_spec(self):
    features_producers = (
        _features_producer(
            {"producer1/feature1": specs.Array(shape=(3,), dtype=np.int32)},
            set(("dacl/measurement1",)),
        ),
        _features_producer(
            {"producer2/feature1": specs.Array(shape=(1,), dtype=np.float32)},
            set(("dacl/measurement1", "producer1/feature1")),
        ),
    )
    dacl_measurements_spec = {
        "dacl/measurement1": specs.BoundedArray(
            shape=(5,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
        "dacl/measurement2": specs.Array(shape=(1,), dtype=np.int32),
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=features_producers,
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    self.assertEqual(
        {
            "dacl/measurement1": specs.BoundedArray(
                shape=(5,), dtype=np.float32, minimum=-1.0, maximum=1.0
            ),
            "dacl/measurement2": specs.Array(shape=(1,), dtype=np.int32),
            "producer1/feature1": specs.Array(shape=(3,), dtype=np.int32),
            "producer2/feature1": specs.Array(shape=(1,), dtype=np.float32),
        },
        task_layer.features_spec(dacl_measurements_spec),
    )

  def test_commands_spec(self):
    commands_processors = (
        _commands_processor(
            {
                "processor1/command1": specs.BoundedArray(
                    shape=(4,), dtype=np.float32, minimum=-12.0, maximum=14.0
                )
            },
            set(("processor2/command1", "dacl/command3")),
        ),
        _commands_processor(
            {"processor2/command1": specs.Array(shape=(1,), dtype=np.float32)},
            set(("dacl/command1", "dacl/command2")),
        ),
    )

    dacl_commands_spec = {
        "dacl/command1": specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-1.0, maximum=1.0
        ),
        "dacl/command2": specs.Array(shape=(3,), dtype=np.float16),
        "dacl/command3": specs.StringArray(shape=()),
        "dacl/command4": specs.DiscreteArray(num_values=4),
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=commands_processors,
        features_producers=(),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    self.assertEqual(
        {
            "processor1/command1": specs.BoundedArray(
                shape=(4,), dtype=np.float32, minimum=-12.0, maximum=14.0
            ),
            "dacl/command4": specs.DiscreteArray(num_values=4),
        },
        task_layer.commands_spec(dacl_commands_spec),
    )

  def test_reward_spec(self):
    reward_provider = mock.create_autospec(
        reaf_reward_provider.RewardProvider, instance=True
    )
    expected_reward_spec = {
        "reward1": specs.Array(shape=(3,), dtype=np.float32),
        "reward2": specs.Array(shape=(3,), dtype=np.float16),
        "reward3": specs.StringArray(shape=()),
    }

    reward_provider.reward_spec.return_value = expected_reward_spec

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=reward_provider,
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )
    self.assertEqual(expected_reward_spec, task_layer.reward_spec())

  def test_discount_spec(self):
    discount_provider = mock.create_autospec(
        reaf_discount_provider.DiscountProvider, instance=True
    )
    expected_discount_spec = {
        "discount1": specs.Array(shape=(3,), dtype=np.float32),
        "discount2": specs.Array(shape=(3,), dtype=np.float16),
        "discount3": specs.StringArray(shape=()),
    }

    discount_provider.discount_spec.return_value = expected_discount_spec

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=discount_provider,
    )
    self.assertEqual(expected_discount_spec, task_layer.discount_spec())

  def test_compute_features(self):
    dacl_measurements = {
        "dacl/obs1": np.array([1.0, -2.0, 12.45]),
        "dacl/obs2": np.array([4, 5, 6], dtype=np.int32),
    }

    producer1_features = {
        "p1/obs1": np.array([1.0, -2.0, 12.45]),
        "p1/obs2": np.array([4, 5, 6], dtype=np.int32),
    }

    def _producer2_features_producer(features):
      dacl_feature = features["dacl/obs1"]
      p1_feature = features["p1/obs2"]
      p1_feature = np.square(p1_feature)
      dacl_feature = dacl_feature + np.array([-0.3, 1.0, -0.45])
      return {"p2/obs1": p1_feature, "p2/obs2": dacl_feature}

    producer1 = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    producer1.produce_features.side_effect = lambda _: producer1_features

    producer2 = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    producer2.produce_features.side_effect = _producer2_features_producer
    producer2.required_features_keys.return_value = set(
        ("dacl/obs1", "p1/obs2")
    )

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(producer1, producer2),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    all_features = task_layer.compute_all_features(
        measurements=dacl_measurements
    )
    np.testing.assert_equal(
        all_features,
        {
            "dacl/obs1": np.array([1.0, -2.0, 12.45]),
            "dacl/obs2": np.array([4, 5, 6], dtype=np.int32),
            "p1/obs1": np.array([1.0, -2.0, 12.45]),
            "p1/obs2": np.array([4, 5, 6], dtype=np.int32),
            "p2/obs1": np.array([16, 25, 36]),
            "p2/obs2": np.array([0.7, -1.0, 12.0]),
        },
    )

  def test_compute_commands(self):
    agent_action = {
        "command1": np.array([1.0, -2.0, 12.45]),
        "command2": np.array([4, 5, 6], dtype=np.int32),
    }

    # Square command2.
    processor1 = mock.create_autospec(
        reaf_commands_processor.CommandsProcessor, instance=True
    )
    processor1.process_commands.side_effect = lambda command: {
        "command2": np.square(command["command2"])
    }

    # We need to specify the spec otherwise the previous command is not
    # forwarded to the processor.
    processor1.consumed_commands_spec.return_value = {
        "command2": specs.Array(shape=(3,), dtype=np.int32)
    }

    # Sum to command1 and change name of the key.
    processor2 = mock.create_autospec(
        reaf_commands_processor.CommandsProcessor, instance=True
    )
    processor2.process_commands.side_effect = lambda command: np.sum(
        {"command3": command["command1"] + np.array([-0.3, 1.0, -0.45])}
    )
    # We need to specify the spec otherwise the previous command is not
    # forwarded to the processor.
    processor2.consumed_commands_spec.return_value = {
        "command1": specs.Array(shape=(3,), dtype=np.float32)
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(processor1, processor2),
        features_producers=(),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    commands = task_layer.compute_final_commands(agent_action)

    np.testing.assert_equal(
        commands,
        {
            "command2": np.array([16, 25, 36]),
            "command3": np.array([0.7, -1.0, 12.0]),
        },
    )

  def test_compute_reward(self):
    features = {
        "feature1": np.array([1.0, -2.0, 12.45]),
        "feature2": np.array([4, 5, 6], dtype=np.int32),
        "feature3": np.array([0.7, -1.0, 12.0]),
    }

    reward_provider = mock.create_autospec(
        reaf_reward_provider.RewardProvider, instance=True
    )
    reward_provider.required_features_keys.return_value = set(
        ("feature1", "feature3")
    )

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=reward_provider,
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    task_layer.compute_reward(features)

    numpy_mock_assertions.assert_called_once_with(
        reward_provider.compute_reward,
        {
            "feature1": np.array([1.0, -2.0, 12.45]),
            "feature3": np.array([0.7, -1.0, 12.0]),
        },
    )

  def test_check_termination_checker_are_called(self):
    features = {
        "feature1": np.array([1.0, -2.0, 12.45]),
        "feature2": np.array([4, 5, 6], dtype=np.int32),
        "feature3": np.array([0.7, -1.0, 12.0]),
    }

    termination_checker1 = mock.create_autospec(
        reaf_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker1",
    )
    termination_checker1.required_features_keys.return_value = set(
        ("feature1", "feature3")
    )

    termination_checker2 = mock.create_autospec(
        reaf_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker2",
    )
    termination_checker2.required_features_keys.return_value = set(
        ("feature2", "feature3")
    )

    termination_state1 = (
        reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE
    )
    termination_state2 = (
        reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE
    )
    termination_checker1.check_termination.return_value = termination_state1
    termination_checker2.check_termination.return_value = termination_state2

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(termination_checker1, termination_checker2),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    task_layer.check_for_termination(features)

    numpy_mock_assertions.assert_called_once_with(
        termination_checker1.check_termination,
        {
            "feature1": np.array([1.0, -2.0, 12.45]),
            "feature3": np.array([0.7, -1.0, 12.0]),
        },
    )
    numpy_mock_assertions.assert_called_once_with(
        termination_checker2.check_termination,
        {
            "feature2": np.array([4, 5, 6], dtype=np.int32),
            "feature3": np.array([0.7, -1.0, 12.0]),
        },
    )

  def test_perform_reset_resets_elements(self):
    termination_checker = mock.create_autospec(
        reaf_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker1",
    )

    producer = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    processor = mock.create_autospec(
        reaf_commands_processor.CommandsProcessor, instance=True
    )
    reward_provider = mock.create_autospec(
        reaf_reward_provider.RewardProvider, instance=True
    )
    discount_provider = mock.create_autospec(
        reaf_discount_provider.DiscountProvider, instance=True
    )

    # Also test when inheriting from two interfaces.
    class _CombinedCommandsProcessorAndFeaturesProducer(
        reaf_commands_processor.CommandsProcessor,
        reaf_features_producer.FeaturesProducer,
    ):
      pass

    producer_processor = mock.create_autospec(
        _CombinedCommandsProcessorAndFeaturesProducer, instance=True
    )

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(processor, producer_processor),
        features_producers=(producer, producer_processor),
        reward_provider=reward_provider,
        termination_checkers=(termination_checker,),
        discount_provider=discount_provider,
    )

    task_layer.perform_reset()

    processor.reset.assert_called_once()
    producer.reset.assert_called_once()
    reward_provider.reset.assert_called_once()
    discount_provider.reset.assert_called_once()
    termination_checker.reset.assert_called_once()
    producer_processor.reset.assert_called_once()

  def test_check_termination_results_are_combined(self):
    features = {
        "feature1": np.array([1.0, -2.0, 12.45]),
        "feature2": np.array([4, 5, 6], dtype=np.int32),
        "feature3": np.array([0.7, -1.0, 12.0]),
    }

    termination_checker1 = mock.create_autospec(
        reaf_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker1",
    )
    termination_checker1.required_features_keys.return_value = set(
        ("feature1", "feature3")
    )

    termination_checker2 = mock.create_autospec(
        reaf_termination_checker.TerminationChecker,
        instance=True,
        name="termination_checker2",
    )
    termination_checker2.required_features_keys.return_value = set(
        ("feature2", "feature3")
    )

    termination_state1 = reaf_termination_checker.TerminationResult.TERMINATE
    termination_state2 = reaf_termination_checker.TerminationResult.TRUNCATE
    termination_checker1.check_termination.return_value = termination_state1
    termination_checker2.check_termination.return_value = termination_state2

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(termination_checker1, termination_checker2),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
    )

    self.assertEqual(
        task_layer.check_for_termination(features),
        termination_state1.combine(termination_state2),
    )

  def test_compute_discount(self):
    features = {
        "feature1": np.array([1.0, -2.0, 12.45]),
        "feature2": np.array([4, 5, 6], dtype=np.int32),
        "feature3": np.array([0.7, -1.0, 12.0]),
    }
    termination_state = reaf_termination_checker.TerminationResult.TERMINATE

    discount_provider = mock.create_autospec(
        reaf_discount_provider.DiscountProvider, instance=True
    )
    discount_provider.required_features_keys.return_value = set(
        ("feature1", "feature3")
    )

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=discount_provider,
    )

    task_layer.compute_discount(features, termination_state)

    numpy_mock_assertions.assert_called_once_with(
        discount_provider.compute_discount,
        {
            "feature1": np.array([1.0, -2.0, 12.45]),
            "feature3": np.array([0.7, -1.0, 12.0]),
        },
        termination_state,
    )

  def test_observers_are_called(self):
    obs1 = mock.create_autospec(
        reaf_features_observers.FeaturesObserver, instance=True
    )
    obs2 = mock.create_autospec(
        reaf_features_observers.FeaturesObserver, instance=True
    )

    dacl_measurements = {
        "dacl/obs1": np.array([1.0, -2.0, 12.45]),
        "dacl/obs2": np.array([4, 5, 6], dtype=np.int32),
    }

    producer1_features = {
        "p1/obs1": np.array([1.0, -2.0, 12.45]),
        "p1/obs2": np.array([4, 5, 6], dtype=np.int32),
    }

    producer2_features = {
        "p2/obs1": np.array([1.7, -1.0, 12.0]),
        "p2/obs2": np.array([16, 25, -36], dtype=np.int32),
    }

    producer1 = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    producer1.produce_features.side_effect = lambda _: producer1_features

    producer2 = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    producer2.produce_features.side_effect = lambda _: producer2_features
    producer2.required_features_keys.return_value = set(
        ("dacl/obs1", "p1/obs2")
    )

    all_features = {
        "dacl/obs1": np.array([1.0, -2.0, 12.45]),
        "dacl/obs2": np.array([4, 5, 6], dtype=np.int32),
        "p1/obs1": np.array([1.0, -2.0, 12.45]),
        "p1/obs2": np.array([4, 5, 6], dtype=np.int32),
        "p2/obs1": np.array([1.7, -1.0, 12.0]),
        "p2/obs2": np.array([16, 25, -36], dtype=np.int32),
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(producer1, producer2),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
        features_observers=(obs1, obs2),
    )

    # Observers are called when computing all the features. Once per call.
    task_layer.compute_all_features(measurements=dacl_measurements)

    numpy_mock_assertions.assert_called_once_with(
        obs1.observe_features,
        all_features,
    )
    numpy_mock_assertions.assert_called_once_with(
        obs2.observe_features,
        all_features,
    )

  def test_loggers_are_called(self):
    logger1 = mock.create_autospec(reaf_logger.Logger, instance=True)
    logger2 = mock.create_autospec(reaf_logger.Logger, instance=True)

    # Add both feature producers and command processors.

    dacl_measurements = {
        "dacl/obs1": np.array([1.0, -2.0, 12.45]),
        "dacl/obs2": np.array([4, 5, 6], dtype=np.int32),
    }

    producer1_features = {
        "p1/obs1": np.array([1.0, -2.0, 12.45]),
        "p1/obs2": np.array([4, 5, 6], dtype=np.int32),
    }

    producer2_features = {
        "p2/obs1": np.array([1.7, -1.0, 12.0]),
        "p2/obs2": np.array([16, 25, -36], dtype=np.int32),
    }

    producer1 = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    producer1.produce_features.side_effect = lambda _: producer1_features

    producer2 = mock.create_autospec(
        reaf_features_producer.FeaturesProducer, instance=True
    )
    producer2.produce_features.side_effect = lambda _: producer2_features
    producer2.required_features_keys.return_value = set(
        ("dacl/obs1", "p1/obs2")
    )

    agent_action = {
        "command1": np.array([1.0, -2.0, 12.45]),
        "command2": np.array([4, 5, 6], dtype=np.int32),
    }

    # Square command2.
    processor1 = mock.create_autospec(
        reaf_commands_processor.CommandsProcessor,
        instance=True,
        name="processor1",
    )
    processor1.process_commands.side_effect = lambda command: {
        "command2": np.square(command["command2"])
    }

    # We need to specify the spec otherwise the previous command is not
    # forwarded to the processor.
    processor1.consumed_commands_spec.return_value = {
        "command2": specs.Array(shape=(3,), dtype=np.int32)
    }

    # Sum to command1 and change name of the key.
    processor2 = mock.create_autospec(
        reaf_commands_processor.CommandsProcessor,
        instance=True,
        name="processor2",
    )

    processor2.process_commands.side_effect = lambda command: np.sum(
        {"command3": command["command1"] + np.array([-0.3, 1.0, -0.45])}
    )
    # We need to specify the spec otherwise the previous command is not
    # forwarded to the processor.
    processor2.consumed_commands_spec.return_value = {
        "command1": specs.Array(shape=(3,), dtype=np.float32)
    }

    all_features = {
        "dacl/obs1": np.array([1.0, -2.0, 12.45]),
        "dacl/obs2": np.array([4, 5, 6], dtype=np.int32),
        "p1/obs1": np.array([1.0, -2.0, 12.45]),
        "p1/obs2": np.array([4, 5, 6], dtype=np.int32),
        "p2/obs1": np.array([1.7, -1.0, 12.0]),
        "p2/obs2": np.array([16, 25, -36], dtype=np.int32),
    }

    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(processor1, processor2),
        features_producers=(producer1, producer2),
        reward_provider=mock.create_autospec(
            reaf_reward_provider.RewardProvider, instance=True
        ),
        termination_checkers=(),
        discount_provider=mock.create_autospec(
            reaf_discount_provider.DiscountProvider, instance=True
        ),
        loggers=(logger1, logger2),
    )

    # Loggers are called when computing features with the dacl measurements and
    # the full features.
    task_layer.compute_all_features(measurements=dacl_measurements)

    numpy_mock_assertions.assert_called_once_with(
        logger1.record_measurements,
        dacl_measurements,
    )
    numpy_mock_assertions.assert_called_once_with(
        logger2.record_measurements,
        dacl_measurements,
    )
    numpy_mock_assertions.assert_called_once_with(
        logger1.record_features,
        all_features,
    )
    numpy_mock_assertions.assert_called_once_with(
        logger2.record_features,
        all_features,
    )

    # Loggers are also called when processing the commands.
    task_layer.compute_final_commands(agent_action)
    expected_calls = (
        mock.call(
            processor1.name,
            {"command2": np.array([4, 5, 6], dtype=np.int32)},
            {"command2": np.array([16, 25, 36])},
        ),
        mock.call(
            processor2.name,
            {"command1": np.array([1.0, -2.0, 12.45])},
            {"command3": np.array([0.7, -1.0, 12.0])},
        ),
    )
    numpy_mock_assertions.assert_has_calls(
        logger1.record_commands_processing, expected_calls
    )
    numpy_mock_assertions.assert_has_calls(
        logger2.record_commands_processing, expected_calls
    )

    # Final command.
    numpy_mock_assertions.assert_called_once_with(
        logger1.record_final_commands,
        {
            "command2": np.array([16, 25, 36]),
            "command3": np.array([0.7, -1.0, 12.0]),
        },
    )
    numpy_mock_assertions.assert_called_once_with(
        logger2.record_final_commands,
        {
            "command2": np.array([16, 25, 36]),
            "command3": np.array([0.7, -1.0, 12.0]),
        },
    )

  def test_default_reward_and_discount_providers(self):
    task_layer = task_logic_layer.TaskLogicLayer(
        commands_processors=(),
        features_producers=(),
        termination_checkers=(),
    )

    self.assertEqual(
        task_layer.reward_spec(),
        # TODO: Fix the correct dtype.
        specs.Array(shape=(1,), dtype=float),
    )
    self.assertEqual(
        task_layer.discount_spec(),
        specs.BoundedArray(
            shape=(),
            dtype=np.float64,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        ),
    )
    # Reset the TTL first.
    task_layer.perform_reset()

    # Arbitrary features.
    features = {
        "feature1": np.array([1.0, -2.0, 12.45]),
    }
    # Check reward is always 0.
    self.assertEqual(task_layer.compute_reward(features), 0.0)

    # Check discount is 1.0 when not terminating.
    self.assertEqual(
        task_layer.compute_discount(
            features,
            reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE,
        ),
        1.0,
    )
    # Check discount is 1.0 when truncating.
    self.assertEqual(
        task_layer.compute_discount(
            features,
            reaf_termination_checker.TerminationResult.TRUNCATE,
        ),
        1.0,
    )
    # Check discount is 0.0 when terminating.
    self.assertEqual(
        task_layer.compute_discount(
            features,
            reaf_termination_checker.TerminationResult.TERMINATE,
        ),
        0.0,
    )


if __name__ == "__main__":
  absltest.main()
