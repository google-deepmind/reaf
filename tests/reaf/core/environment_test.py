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
"""Tests for REAF environment."""

from collections.abc import Callable, Mapping
import dataclasses
from unittest import mock

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import types as gdmr_types
from gdm_robotics.testing import specs_utils as test_specs
import numpy as np
from reaf.core import action_space_adapter as reaf_action_space_adapter
from reaf.core import data_acquisition_and_control_layer as reaf_dacl
from reaf.core import environment as reaf_environment
from reaf.core import logger as reaf_logger
from reaf.core import numpy_mock_assertions
from reaf.core import observation_space_adapter as reaf_observation_space_adapter
from reaf.core import task_logic_layer as reaf_tll
from reaf.core import termination_checker as reaf_termination_checker
from reaf.testing import fakes
import tree
from typing_extensions import override

from absl.testing import absltest
from absl.testing import parameterized


class EnvironmentTest(parameterized.TestCase):

  def test_action_spec_corresponds_to_action_adapter(self):
    expected_action_spec = {
        "action1": specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-18.0, maximum=19.0
        ),
        "action2": specs.Array(
            shape=(1,),
            dtype=np.int32,
        ),
        "action3": specs.StringArray(shape=()),
    }

    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.action_spec.return_value = expected_action_spec

    ttl = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match.
    ttl.commands_spec.return_value = expected_action_spec
    action_adapter.task_commands_keys.return_value = set((
        "action1",
        "action2",
        "action3",
    ))

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
    )
    self.assertEqual(expected_action_spec, environment.action_spec())

  def test_timestep_spec_is_assembled_correctly(self):
    # Timestep spec is composed of different parts: observation, reward and
    # discount.
    expected_observation_spec = {
        "observation1": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=-18.0, maximum=19.0
        ),
        "observation2": specs.Array(shape=(1,), dtype=np.int32),
    }
    observation_adapter = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    observation_adapter.observation_spec.return_value = (
        expected_observation_spec
    )

    ttl = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    expected_reward_spec = {
        "main_reward": specs.BoundedArray(
            shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
        ),
    }

    expected_discount_spec = {
        "discount": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
        ),
    }

    ttl.reward_spec.return_value = expected_reward_spec
    ttl.discount_spec.return_value = expected_discount_spec

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl,
        action_space_adapter=action_adapter,
        observation_space_adapter=observation_adapter,
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
    )
    self.assertEqual(
        gdmr_types.TimeStepSpec(
            step_type=gdmr_types.STEP_TYPE_SPEC,
            reward=expected_reward_spec,
            discount=expected_discount_spec,
            observation=expected_observation_spec,
        ),
        environment.timestep_spec(),
    )

  def test_missing_action_adapter_keys_raises_error(self):
    task_commands_spec = {
        "action1": specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-18.0, maximum=19.0
        ),
        "action2": specs.Array(
            shape=(1,),
            dtype=np.int32,
        ),
        "action3": specs.StringArray(shape=()),
    }

    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set((
        "action2",
        "action4",
    ))

    ttl = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl.commands_spec.return_value = task_commands_spec

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    with self.assertRaises(ValueError):
      reaf_environment.Environment(
          data_acquisition_and_control_layer=mock.create_autospec(
              reaf_dacl.DataAcquisitionAndControlLayer, instance=True
          ),
          task_logic_layer=ttl,
          action_space_adapter=action_adapter,
          observation_space_adapter=mock.create_autospec(
              reaf_observation_space_adapter.ObservationSpaceAdapter,
              instance=True,
          ),
          environment_reset=mock.create_autospec(
              reaf_environment.EnvironmentReset, instance=True
          ),
          end_of_episode_handler=None,
      )

  def test_missing_observation_adapter_keys_raises_error(self):
    task_features_spec = {
        "observation1": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=-18.0, maximum=19.0
        ),
        "observation2": specs.Array(shape=(1,), dtype=np.int32),
    }
    observation_adapter = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )

    observation_adapter.task_features_keys.return_value = set((
        "observation1",
        "observation3",
    ))

    ttl = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl.features_spec.return_value = task_features_spec
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    with self.assertRaises(ValueError):
      reaf_environment.Environment(
          data_acquisition_and_control_layer=mock.create_autospec(
              reaf_dacl.DataAcquisitionAndControlLayer, instance=True
          ),
          task_logic_layer=ttl,
          action_space_adapter=action_adapter,
          observation_space_adapter=observation_adapter,
          environment_reset=mock.create_autospec(
              reaf_environment.EnvironmentReset, instance=True
          ),
          end_of_episode_handler=None,
      )

  def test_reset_function_is_called(self):
    # Define the custom options class. This must be a dataclass and inherit from
    # gdmr_env.Options.

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    # Not sure why we need to create a mock for the method instead of using the
    # generic mock returned above, but without this tests will fail with a
    # "TypeError: missing a required argument: 'config'" error.
    environment_reset.do_reset = mock.MagicMock()

    initial_obs = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }

    all_features = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32),
    }
    final_observation = {
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32)
    }

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    dacl_mock.begin_stepping.return_value = initial_obs

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.compute_all_features.return_value = all_features
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    observation_adapter_mock = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    observation_adapter_mock.observations_from_features.return_value = (
        final_observation
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=observation_adapter_mock,
        environment_reset=environment_reset,
        end_of_episode_handler=None,
    )

    reset_timestep = environment.reset_with_options(
        options=ResetOptions(option1="my_option1", option2=42)
    )

    # Check equality of the timestep.
    expected_timestep = dm_env.TimeStep(
        step_type=np.array([dm_env.StepType.FIRST]),
        # Reward and discount must be zero matching the spec.
        reward=np.zeros(2, dtype=np.float32),
        discount=np.zeros(1, dtype=np.float32),
        observation=final_observation,
    )

    np.testing.assert_equal(reset_timestep, expected_timestep)

    # Check correct methods have been called.
    environment_reset.do_reset.assert_called_once_with(
        ResetOptions(option1="my_option1", option2=42),
    )

    # Note: this is invisible from the public API, but we assert as with simple
    # mocks we do not enforce that the output has gone through the correct
    # pipeline.
    numpy_mock_assertions.assert_called_once_with(
        ttl_mock.compute_all_features, initial_obs
    )
    numpy_mock_assertions.assert_called_once_with(
        observation_adapter_mock.observations_from_features, all_features
    )

  def test_reset_function_can_be_overridden(self):
    # Define the custom options class. This must be a dataclass and inherit from
    # gdmr_env.Options.

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    original_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    # Not sure why we need to create a mock for the method instead of using the
    # generic mock returned above, but without this tests will fail with a
    # "TypeError: missing a required argument: 'config'" error.
    original_reset.do_reset = mock.MagicMock()
    environment_reset.do_reset = mock.MagicMock()

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.compute_all_features.return_value = {}
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.Array(
        shape=(1,), dtype=np.float32
    )

    ttl_mock.discount_spec.return_value = specs.Array(
        shape=(1,), dtype=np.float32
    )

    observation_adapter_mock = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=observation_adapter_mock,
        environment_reset=original_reset,
        end_of_episode_handler=None,
    )

    # Override the reset function after construction.
    environment.environment_reset = environment_reset

    environment.reset_with_options(
        options=ResetOptions(option1="my_option1", option2=42)
    )

    # Check that the new reset function is called, and not the original one.
    environment_reset.do_reset.assert_called_once_with(
        ResetOptions(option1="my_option1", option2=42),
    )
    original_reset.do_reset.assert_not_called()

  def test_reset_resets_task_logic_layer(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      pass

    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    # Not sure why we need to create a mock for the method instead of using the
    # generic mock returned above, but without this tests will fail with a
    # "TypeError: missing a required argument: 'config'" error.
    environment_reset.do_reset = mock.MagicMock()

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.compute_all_features.return_value = {}
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.Array(
        shape=(1,), dtype=np.float32
    )

    ttl_mock.discount_spec.return_value = specs.Array(
        shape=(1,), dtype=np.float32
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=environment_reset,
        end_of_episode_handler=None,
    )

    environment.reset_with_options(options=ResetOptions())
    ttl_mock.perform_reset.assert_called_once()

  def test_end_of_episode_handler_is_called_on_last_timestep(self):

    end_of_episode_handler = mock.create_autospec(
        reaf_environment.EndOfEpisodeHandler, instance=True
    )

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    dacl_mock.begin_stepping.return_value = {}

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.compute_all_features.return_value = {}
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # We need to trigger a termination so that the timestep will be marked as
    # `LAST`
    ttl_mock.check_for_termination.return_value = (
        reaf_termination_checker.TerminationResult.TERMINATE
    )

    observation_adapter_mock = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    observation_adapter_mock.observations_from_features.return_value = {}

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=observation_adapter_mock,
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=end_of_episode_handler,
        action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.IGNORE,
    )
    environment.reset()
    environment.step({})
    environment.reset()

    # Check correct method has been called.
    end_of_episode_handler.on_end_of_episode_stepping.assert_called_once()

  def test_explicit_default_reset_configuration_is_correct(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    environment_reset.default_reset_configuration = mock.MagicMock(
        return_value=ResetOptions(option1="my_option1", option2=42)
    )
    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=environment_reset,
        end_of_episode_handler=None,
    )
    self.assertEqual(
        ResetOptions(option1="my_option1", option2=42),
        environment.default_reset_options(),
    )

  def test_default_reset_configuration(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    # We need to test a default implementation of the abstract class. We use a
    # concrete object instead of a mock.
    class _MyEnvironmentReset(reaf_environment.EnvironmentReset[ResetOptions]):

      def do_reset(self, options: ResetOptions) -> None:
        del options

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=_MyEnvironmentReset(),
        end_of_episode_handler=None,
    )
    self.assertEqual(
        gdmr_env.Options(),
        environment.default_reset_options(),
    )

  def test_step_on_fresh_environment_triggers_reset(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    # Not sure why we need to create a mock for the method instead of using the
    # generic mock returned above, but without this tests will fail with a
    # "TypeError: missing a required argument: 'config'" error.
    environment_reset.do_reset = mock.MagicMock()
    environment_reset.default_reset_configuration = mock.MagicMock(
        return_value=ResetOptions(option1="my_option1", option2=42)
    )

    initial_obs = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }

    all_features = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32),
    }
    final_observation = {
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32)
    }

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    dacl_mock.begin_stepping.return_value = initial_obs

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.compute_all_features.return_value = all_features

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    observation_adapter_mock = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    observation_adapter_mock.observations_from_features.return_value = (
        final_observation
    )

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=observation_adapter_mock,
        environment_reset=environment_reset,
        end_of_episode_handler=None,
    )

    # Arbitrary action as it will be ignored anyway.
    agent_action = np.array([0.23, -0.65, 0.3]).astype(np.float32)
    timestep = environment.step(agent_action)
    # Check equality of the timestep. It must be a first step.
    expected_timestep = dm_env.TimeStep(
        step_type=np.array([dm_env.StepType.FIRST]),
        # Reward and discount must be zero matching the spec.
        reward=np.zeros(2, dtype=np.float32),
        discount=np.zeros(1, dtype=np.float32),
        observation=final_observation,
    )
    np.testing.assert_equal(timestep, expected_timestep)

    # Check correct methods have been called. Note that environment will call
    # `reset` which in turn will call the reset function with the default
    # option.
    environment_reset.do_reset.assert_called_once_with(
        ResetOptions(option1="my_option1", option2=42),
    )

    # Check that the action adapter has not been called.
    action_adapter.commands_from_environment_action.assert_not_called()

  def test_step_on_terminated_episode_triggers_reset(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    # Not sure why we need to create a mock for the method instead of using the
    # generic mock returned above, but without this tests will fail with a
    # "TypeError: missing a required argument: 'config'" error.
    environment_reset.do_reset = mock.MagicMock()
    environment_reset.default_reset_configuration = mock.MagicMock(
        return_value=ResetOptions(option1="my_option1", option2=42)
    )

    initial_obs = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }

    all_features = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32),
    }
    final_observation = {
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32)
    }

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    dacl_mock.begin_stepping.return_value = initial_obs

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.compute_all_features.return_value = all_features

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    action_adapter_mock = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    ttl_mock.commands_spec.return_value = {}
    action_adapter_mock.task_commands_keys.return_value = set()

    observation_adapter_mock = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    observation_adapter_mock.observations_from_features.return_value = (
        final_observation
    )

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter_mock,
        observation_space_adapter=observation_adapter_mock,
        environment_reset=environment_reset,
        end_of_episode_handler=None,
    )

    # Build a timestep that has step_type=LAST. We do not care about reward and
    # observation, so the spec do not really match, and we can use the wrong
    # dm_env utility function.
    last_timestep = dm_env.termination(
        reward=np.zeros(1, dtype=np.float32), observation={}
    )

    with mock.patch.object(environment, "_last_timestep", last_timestep):
      # Arbitrary action as it will be ignored anyway.
      agent_action = np.array([0.23, -0.65, 0.3]).astype(np.float32)
      timestep = environment.step(agent_action)
      # Check equality of the timestep. It must be a first step.
      expected_timestep = dm_env.TimeStep(
          step_type=np.array([dm_env.StepType.FIRST]),
          # Reward and discount must be zero matching the spec.
          reward=np.zeros(2, dtype=np.float32),
          discount=np.zeros(1, dtype=np.float32),
          observation=final_observation,
      )
      np.testing.assert_equal(timestep, expected_timestep)

      # Check correct methods have been called. Note that environment will call
      # `reset` which in turn will call the reset function with the default
      # option.
      environment_reset.do_reset.assert_called_once_with(
          ResetOptions(option1="my_option1", option2=42),
      )

      # Check that the action adapter has not been called.
      action_adapter_mock.commands_from_environment_action.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name="no_termination",
          termination_result=reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE,
          timestep_builder=lambda obs, reward, discount: dm_env.TimeStep(
              step_type=np.array([dm_env.StepType.MID]),
              reward=reward,
              discount=discount,
              observation=obs,
          ),
      ),
      dict(
          testcase_name="termination",
          termination_result=reaf_termination_checker.TerminationResult.TERMINATE,
          timestep_builder=lambda obs, reward, _: dm_env.TimeStep(
              step_type=np.array([dm_env.StepType.LAST]),
              reward=reward,
              discount={"discount": np.zeros(1, dtype=np.float32)},
              observation=obs,
          ),
      ),
      dict(
          testcase_name="truncation",
          termination_result=reaf_termination_checker.TerminationResult.TRUNCATE,
          timestep_builder=lambda obs, reward, discount: dm_env.TimeStep(
              step_type=np.array([dm_env.StepType.LAST]),
              reward=reward,
              discount=discount,
              observation=obs,
          ),
      ),
  )
  def test_step_function(
      self,
      termination_result: reaf_termination_checker.TerminationResult,
      timestep_builder: Callable[
          [
              tree.Structure[gdmr_types.ArrayType],  # Observation.
              tree.Structure[gdmr_types.ArrayType],  # Reward.
              tree.Structure[gdmr_types.ArrayType],  # Discount.
          ],
          dm_env.TimeStep,
      ],
  ):
    agent_action = np.array([0.23, -0.65, 0.3]).astype(np.float32)
    action_adapter_output = {
        "action1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "action2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }

    action_adapter_mock = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter_mock.commands_from_environment_action.return_value = (
        action_adapter_output
    )

    initial_obs = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }

    all_features = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32),
    }
    final_observation = {
        "feature1": np.array([0.13, 0.52, -0.3]).astype(np.float32)
    }

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    dacl_mock.step.return_value = initial_obs

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)

    # Action adapter and task commands spec must match.
    action_adapter_mock.task_commands_keys.return_value = set(
        ("action1", "action2")
    )
    ttl_mock.commands_spec.return_value = {
        "action1": specs.Array(shape=(3,), dtype=np.float32),
        "action2": specs.Array(shape=(5,), dtype=np.int32),
    }

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = {
        "reward1": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=0.0, maximum=100.0
        ),
    }
    ttl_mock.discount_spec.return_value = {
        "discount": specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
        ),
    }
    ttl_mock.compute_all_features.return_value = all_features
    ttl_mock.compute_reward.return_value = {
        "reward1": np.asarray(5.6).astype(np.float32)
    }
    ttl_mock.compute_discount.return_value = {
        "discount": np.asarray(0.9).astype(np.float32)
    }
    ttl_mock.check_for_termination.return_value = termination_result

    observation_adapter_mock = mock.create_autospec(
        reaf_observation_space_adapter.ObservationSpaceAdapter, instance=True
    )
    observation_adapter_mock.observations_from_features.return_value = (
        final_observation
    )

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter_mock,
        observation_space_adapter=observation_adapter_mock,
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
        action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.IGNORE,
    )
    # Reset the environment first.
    environment.reset()

    # Reset mocks state after reset.
    ttl_mock.compute_all_features.reset_mock()
    observation_adapter_mock.observations_from_features.reset_mock()

    timestep = environment.step(agent_action)
    # Check equality of the timestep.
    expected_timestep = timestep_builder(
        final_observation,
        {"reward1": np.asarray(5.6).astype(np.float32)},
        {"discount": np.asarray(0.9).astype(np.float32)},
    )
    np.testing.assert_equal(timestep, expected_timestep)

    # Note: this is invisible from the public API, but we assert as with simple
    # mocks we do not enforce that the output has gone through the correct
    # pipeline.

    # Input pipeline.
    action_adapter_mock.commands_from_environment_action.assert_called_once_with(
        agent_action
    )
    ttl_mock.compute_final_commands.assert_called_once_with(
        action_adapter_output
    )

    dacl_mock.step.assert_called_once()

    # Output pipeline.
    numpy_mock_assertions.assert_called_once_with(
        ttl_mock.compute_all_features, initial_obs
    )
    numpy_mock_assertions.assert_called_once_with(
        ttl_mock.compute_reward, all_features
    )
    numpy_mock_assertions.assert_called_once_with(
        ttl_mock.check_for_termination, all_features
    )
    numpy_mock_assertions.assert_called_once_with(
        ttl_mock.compute_discount,
        all_features,
        termination_result,
    )
    numpy_mock_assertions.assert_called_once_with(
        observation_adapter_mock.observations_from_features, all_features
    )

  def test_reset_triggers_dacl_begin_stepping(self):
    # Define the custom options class. This must be a dataclass and inherit from
    # gdmr_env.Options.

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class ResetOptions(gdmr_env.Options):
      option1: str
      option2: int

    environment_reset = mock.create_autospec(
        reaf_environment.EnvironmentReset[ResetOptions], instance=True
    )
    # Not sure why we need to create a mock for the method instead of using the
    # generic mock returned above, but without this tests will fail with a
    # "TypeError: missing a required argument: 'config'" error.
    environment_reset.do_reset = mock.MagicMock()

    ttl = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()
    ttl.commands_spec.return_value = {}

    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=environment_reset,
        end_of_episode_handler=None,
    )

    environment.reset_with_options(
        options=ResetOptions(option1="my_option1", option2=42)
    )
    dacl_mock.begin_stepping.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name="no_termination",
          termination_result=reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE,
          should_assert=False,
      ),
      dict(
          testcase_name="termination",
          termination_result=reaf_termination_checker.TerminationResult.TERMINATE,
          should_assert=True,
      ),
      dict(
          testcase_name="truncation",
          termination_result=reaf_termination_checker.TerminationResult.TRUNCATE,
          should_assert=True,
      ),
  )
  def test_termination_triggers_dacl_end_stepping(
      self,
      termination_result: reaf_termination_checker.TerminationResult,
      should_assert: bool,
  ):
    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.check_for_termination.return_value = termination_result

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()
    ttl_mock.commands_spec.return_value = {}

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
        action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.IGNORE,
    )

    # Change the timestep to be different than the default None so that we can
    # call `step`. We only set the step_type.
    with mock.patch.object(
        environment,
        "_last_timestep",
        dm_env.TimeStep(
            step_type=np.asarray(dm_env.StepType.MID),
            reward=np.asarray(0.0),
            discount=np.asarray(0.0),
            observation={},
        ),
    ):
      environment.step({})

    if should_assert:
      dacl_mock.end_stepping.assert_called_once()
    else:
      dacl_mock.end_stepping.assert_not_called()

  def test_early_reset_triggers_dacl_end_stepping_and_end_of_episode_handler(
      self,
  ):
    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.check_for_termination.return_value = (
        reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE
    )

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    action_adapter_mock = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter_mock.task_commands_keys.return_value = set()
    ttl_mock.commands_spec.return_value = {}

    end_of_episode_handler_mock = mock.create_autospec(
        reaf_environment.EndOfEpisodeHandler, instance=True
    )

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter_mock,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=end_of_episode_handler_mock,
        action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.IGNORE,
    )

    environment.step({})
    environment.reset()

    dacl_mock.end_stepping.assert_called_once()
    end_of_episode_handler_mock.on_end_of_episode_stepping.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name="flat_spec",
          reward_spec=specs.BoundedArray(
              shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
          ),
          discount_spec=specs.BoundedArray(
              shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
          ),
          expected_zero_reward=np.zeros((2,), dtype=np.float32),
          expected_zero_discount=np.zeros((1,), dtype=np.float32),
      ),
      dict(
          testcase_name="dict_like_spec",
          reward_spec={
              "reward1": specs.BoundedArray(
                  shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
              ),
              "reward2": specs.BoundedArray(
                  shape=(4,), dtype=np.int32, minimum=0, maximum=5
              ),
          },
          discount_spec={
              "discount1": specs.BoundedArray(
                  shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
              ),
              "discount2": specs.Array(shape=(2,), dtype=np.int16),
          },
          expected_zero_reward={
              "reward1": np.zeros((2,), dtype=np.float32),
              "reward2": np.zeros((4,), dtype=np.int32),
          },
          expected_zero_discount={
              "discount1": np.zeros((1,), dtype=np.float32),
              "discount2": np.zeros((2,), dtype=np.int16),
          },
      ),
  )
  def test_zero_discount_and_reward_match_specs(
      self,
      reward_spec: gdmr_types.RewardSpec,
      discount_spec: gdmr_types.DiscountSpec,
      expected_zero_reward: tree.Structure[gdmr_types.ArrayType],
      expected_zero_discount: tree.Structure[gdmr_types.ArrayType],
  ):
    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.reward_spec.return_value = reward_spec
    ttl_mock.discount_spec.return_value = discount_spec

    # Action adapter and task commands spec must match. As we do not use
    # directly the values in this test, we just make the return value
    # compatible.
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()
    ttl_mock.commands_spec.return_value = {}

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
    )

    # We test only reset as it will return both zero reward and discount.
    timestep = environment.reset()
    np.testing.assert_equal(timestep.reward, expected_zero_reward)
    np.testing.assert_equal(timestep.discount, expected_zero_discount)

  def test_returns_dacl_and_task_logic(self):
    # Create fake objects to make an environment.
    # Choice is arbitrary.
    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(2,), dtype=np.float32, minimum=0.0, maximum=100.0
    )
    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )
    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.task_commands_keys.return_value = set()

    # Make environment, and ensure it returns the same objects passed during
    # construction.
    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
    )
    self.assertEqual(dacl_mock, environment.data_acquisition_and_control_layer)
    self.assertEqual(ttl_mock, environment.task_logic_layer)

  @parameterized.named_parameters(
      dict(
          testcase_name="raise_error",
          action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.RAISE_ERROR,
          valid_action=np.array([0.0, 0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape=np.array([0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape_should_raise_error=True,
          invalid_action_wrong_dtype=np.array(
              [0.0, 0.0, 0.0], dtype=np.float64
          ),
          invalid_action_wrong_dtype_should_raise_error=True,
          invalid_action_out_of_bounds=np.array(
              [100.0, 0.0, 0.0], dtype=np.float32
          ),
          invalid_action_out_of_bounds_should_raise_error=True,
      ),
      dict(
          testcase_name="ignore",
          action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.IGNORE,
          valid_action=np.array([0.0, 0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape=np.array([0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape_should_raise_error=False,
          invalid_action_wrong_dtype=np.array(
              [0.0, 0.0, 0.0], dtype=np.float64
          ),
          invalid_action_wrong_dtype_should_raise_error=False,
          invalid_action_out_of_bounds=np.array(
              [100.0, 0.0, 0.0], dtype=np.float32
          ),
          invalid_action_out_of_bounds_should_raise_error=False,
      ),
      dict(
          testcase_name="clip_to_spec",
          action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.CLIP_TO_SPEC,
          valid_action=np.array([0.0, 0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape=None,
          invalid_action_wrong_shape_should_raise_error=True,
          invalid_action_wrong_dtype=None,
          invalid_action_wrong_dtype_should_raise_error=False,
          invalid_action_out_of_bounds=np.array(
              [100.0, 0.0, 0.0], dtype=np.float32
          ),
          invalid_action_out_of_bounds_should_raise_error=False,
      ),
      dict(
          testcase_name="warning",
          action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.WARNING,
          valid_action=np.array([0.0, 0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape=np.array([0.0, 0.0], dtype=np.float32),
          invalid_action_wrong_shape_should_raise_error=False,
          invalid_action_wrong_dtype=np.array(
              [0.0, 0.0, 0.0], dtype=np.float64
          ),
          invalid_action_wrong_dtype_should_raise_error=False,
          invalid_action_out_of_bounds=np.array(
              [100.0, 0.0, 0.0], dtype=np.float32
          ),
          invalid_action_out_of_bounds_should_raise_error=False,
      ),
  )
  def test_action_spec_enforcement_options(
      self,
      action_spec_enforcement_option: reaf_environment.ActionSpecEnforcementOption,
      valid_action: np.ndarray,
      invalid_action_wrong_shape: np.ndarray | None,
      invalid_action_wrong_shape_should_raise_error: bool,
      invalid_action_wrong_dtype: np.ndarray | None,
      invalid_action_wrong_dtype_should_raise_error: bool,
      invalid_action_out_of_bounds: np.ndarray | None,
      invalid_action_out_of_bounds_should_raise_error: bool,
  ):
    valid_action_spec = specs.BoundedArray(
        shape=(3,), dtype=np.float32, minimum=-18.0, maximum=19.0
    )

    action_adapter = mock.create_autospec(
        reaf_action_space_adapter.ActionSpaceAdapter, instance=True
    )
    action_adapter.action_spec.return_value = valid_action_spec

    ttl = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl.reward_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )

    # Action adapter and task commands spec must match.
    ttl.commands_spec.return_value = {"action1": valid_action_spec}
    action_adapter.task_commands_keys.return_value = set(("action1",))
    recorded_actions = []
    action_adapter.commands_from_environment_action.side_effect = (
        recorded_actions.append
    )

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=mock.create_autospec(
            reaf_dacl.DataAcquisitionAndControlLayer, instance=True
        ),
        task_logic_layer=ttl,
        action_space_adapter=action_adapter,
        observation_space_adapter=mock.create_autospec(
            reaf_observation_space_adapter.ObservationSpaceAdapter,
            instance=True,
        ),
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        end_of_episode_handler=None,
        action_spec_enforcement_option=action_spec_enforcement_option,
    )

    environment.reset()
    # Valid action should not raise an error.
    environment.step(valid_action)
    np.testing.assert_array_equal(recorded_actions[0], valid_action)

    if (
        invalid_action_wrong_shape is not None
        and invalid_action_wrong_shape_should_raise_error
    ):
      with self.subTest("wrong_shape"):
        with self.assertRaises(ValueError):
          environment.step(invalid_action_wrong_shape)

    if (
        invalid_action_wrong_dtype is not None
        and invalid_action_wrong_dtype_should_raise_error
    ):
      with self.subTest("wrong_dtype"):
        with self.assertRaises(ValueError):
          environment.step(invalid_action_wrong_dtype)

    if invalid_action_out_of_bounds is not None:
      with self.subTest("out_of_bounds_value"):
        if invalid_action_out_of_bounds_should_raise_error:
          with self.assertRaises(ValueError):
            environment.step(invalid_action_out_of_bounds)
        elif (
            action_spec_enforcement_option
            == reaf_environment.ActionSpecEnforcementOption.CLIP_TO_SPEC
        ):
          # check clipping.
          recorded_actions = []
          action_adapter.commands_from_environment_action.side_effect = (
              recorded_actions.append
          )
          environment.reset()
          environment.step(invalid_action_out_of_bounds)
          np.testing.assert_array_equal(
              recorded_actions[0],
              np.clip(
                  invalid_action_out_of_bounds,
                  valid_action_spec.minimum,
                  valid_action_spec.maximum,
              ),
          )
        elif (
            action_spec_enforcement_option
            == reaf_environment.ActionSpecEnforcementOption.WARNING
        ):
          # check that no error is raised and warning is logged.
          with self.assertLogs(level="WARNING") as log_output:
            environment.step(invalid_action_out_of_bounds)
          self.assertIn(
              "Failed to validate action against spec", log_output.output[0]
          )

  def test_default_adapters(self):
    """Tests that the default adapters are used."""
    dacl_output = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }

    # DACL values are needed as the environment will use them internally. For
    # this test specifically we only care about the TTL.
    dacl_mock = mock.create_autospec(
        reaf_dacl.DataAcquisitionAndControlLayer, instance=True
    )
    dacl_mock.step.return_value = dacl_output
    dacl_mock.measurements_spec.return_value = {
        "observation1": specs.Array(shape=(3,), dtype=np.float32),
        "observation2": specs.Array(shape=(5,), dtype=np.int32),
    }
    dacl_mock.commands_spec.return_value = {
        "action1": specs.Array(shape=(3,), dtype=np.float32),
        "action2": specs.Array(shape=(5,), dtype=np.int32),
    }

    ttl_mock = mock.create_autospec(reaf_tll.TaskLogicLayer, instance=True)
    ttl_mock.commands_spec.return_value = {
        "action1": specs.Array(shape=(3,), dtype=np.float32),
        "action2": specs.Array(shape=(5,), dtype=np.int32),
    }
    ttl_mock.features_spec.return_value = {
        "observation1": specs.Array(shape=(3,), dtype=np.float32),
        "observation2": specs.Array(shape=(5,), dtype=np.int32),
        "feature1": specs.Array(shape=(3,), dtype=np.float32),
    }

    # Define reward and discount spec as the environment internally creates zero
    # reward and discount based on the spec.
    # Choice here is arbitrary.
    ttl_mock.reward_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0.0, maximum=100.0
    )

    ttl_mock.discount_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
    )
    ttl_mock.compute_all_features.return_value = {
        "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
        "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
        "feature1": np.array([0.1, 0.2, 0.3]).astype(np.float32),
    }

    environment = reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl_mock,
        task_logic_layer=ttl_mock,
        environment_reset=mock.create_autospec(
            reaf_environment.EnvironmentReset, instance=True
        ),
        action_spec_enforcement_option=reaf_environment.ActionSpecEnforcementOption.IGNORE,
    )

    # This should match the TTL commands spec.
    self.assertEqual(
        environment.action_spec(),
        {
            "action1": specs.Array(shape=(3,), dtype=np.float32),
            "action2": specs.Array(shape=(5,), dtype=np.int32),
        },
    )
    # This should match the TTL reward, discount spec and the features spec
    expected_timestep_spec = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=0.0, maximum=100.0
        ),
        discount=specs.BoundedArray(
            shape=(1,), dtype=np.float32, minimum=0, maximum=1.0
        ),
        observation={
            "observation1": specs.Array(shape=(3,), dtype=np.float32),
            "observation2": specs.Array(shape=(5,), dtype=np.int32),
            "feature1": specs.Array(shape=(3,), dtype=np.float32),
        },
    )
    self.assertEqual(environment.timestep_spec(), expected_timestep_spec)

    # Check the actual values.

    # Reset the environment first.
    reset_timestep = environment.reset()
    np.testing.assert_equal(
        reset_timestep.observation,
        {
            "observation1": np.array([0.23, -0.42, 0.3]).astype(np.float32),
            "observation2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
            "feature1": np.array([0.1, 0.2, 0.3]).astype(np.float32),
        },
    )
    agent_action = {
        "action1": np.array([0.1, 0.2, 0.3]).astype(np.float32),
        "action2": np.array([1, 2, -4, 5, -19]).astype(np.int32),
    }
    environment.step(agent_action)
    numpy_mock_assertions.assert_called_once_with(
        ttl_mock.compute_final_commands, agent_action
    )


class SpyLogger(reaf_logger.Logger):

  def __init__(self):
    self.methods_called = []

  @override
  @property
  def name(self) -> str:
    return "LifecycleTestLogger"

  @override
  def record_measurements(
      self, measurements: Mapping[str, gdmr_types.ArrayType]
  ) -> None:
    self.methods_called.append("record_measurements")

  @override
  def record_features(
      self, features: Mapping[str, gdmr_types.ArrayType]
  ) -> None:
    self.methods_called.append("record_features")

  @override
  def record_final_commands(
      self, commands: Mapping[str, gdmr_types.ArrayType]
  ) -> None:
    self.methods_called.append("record_final_commands")

  @override
  def record_commands_processing(
      self,
      name: str,
      consumed_commands: Mapping[str, gdmr_types.ArrayType],
      produced_commands: Mapping[str, gdmr_types.ArrayType],
  ) -> None:
    self.methods_called.append("record_commands_processing")


class EnvironmentLoggerIntegrationTest(absltest.TestCase):

  def _create_environment(self) -> reaf_environment.Environment:
    observation_name = "measurement"

    device = fakes.FakeDevice.one_command_one_measurement(
        "device", "command", (2,), observation_name, (3,)
    )
    dacl = fakes.create_dacl(device)
    command_processor = fakes.CommandRenamer(
        {"command": "agent_command"}, dacl.commands_spec()
    )

    tll = fakes.create_task_layer(
        commands_processors=[command_processor],
    )
    action_spec = test_specs.random_array_spec(shape=(4,))
    action_adapter = fakes.prefixing_action_adapter(
        "agent_command", action_spec
    )
    features_spec = dacl.measurements_spec()
    observation_adapter = fakes.no_op_observation_adapter(features_spec)

    return reaf_environment.Environment(
        data_acquisition_and_control_layer=dacl,
        task_logic_layer=tll,
        action_space_adapter=action_adapter,
        observation_space_adapter=observation_adapter,
        environment_reset=fakes.no_op_reset(),
        end_of_episode_handler=reaf_environment.EndOfEpisodeHandler(),
    )

  def test_lifecycle(self):
    """Tests that the logger functions are called in the correct order."""
    environment = self._create_environment()

    logger = mock.create_autospec(reaf_logger.Logger, instance=True)
    environment.add_logger(logger)

    def logger_functions_called() -> list[str]:
      return [call[0] for call in logger.mock_calls]

    self.assertEqual(logger_functions_called(), [])

    environment.reset()
    self.assertEqual(
        logger_functions_called(),
        [
            "record_measurements",
            "record_features",
        ],
    )

    environment.step(
        test_specs.valid_primitive_value(environment.action_spec())
    )
    self.assertEqual(
        logger_functions_called(),
        [
            "record_measurements",
            "record_features",
            "record_commands_processing",
            "record_final_commands",
            "record_measurements",
            "record_features",
        ],
    )


if __name__ == "__main__":
  absltest.main()
