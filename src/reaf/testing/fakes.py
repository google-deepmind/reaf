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
"""Fake, no-op, spy or trivial implementations of REAF components."""

from collections.abc import Mapping, Sequence
import random

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from gdm_robotics.testing import specs_utils as test_specs
import numpy as np
from reaf.common import environment_reset_from_callable
from reaf.common import partitioner_action_space_adapter
from reaf.core import action_space_adapter
from reaf.core import commands_processor as reaf_commands_processor
from reaf.core import data_acquisition_and_control_layer as reaf_dacl
from reaf.core import default_discount_provider
from reaf.core import default_observation_space_adapter
from reaf.core import device
from reaf.core import device_coordinator
from reaf.core import discount_provider as reaf_discount_provider
from reaf.core import environment
from reaf.core import features_observer as reaf_features_observers
from reaf.core import features_producer as reaf_features_producer
from reaf.core import logger as reaf_logger
from reaf.core import observation_space_adapter
from reaf.core import reward_provider as reaf_reward_provider
from reaf.core import task_logic_layer as reaf_tll
from reaf.core import termination_checker as reaf_termination_checker
import tree
from typing_extensions import override


def create_dacl(
    *devices: device.Device,
) -> reaf_dacl.DataAcquisitionAndControlLayer:
  """Returns a DataAcquisitionAndControlLayer.

  Args:
    *devices: devices to be used, a FakeDeviceCoordinator is given to the DACL.
  """

  return reaf_dacl.DataAcquisitionAndControlLayer(
      device_coordinator=FakeDeviceCoordinator(devices=list(devices)),
      commands_trigger=None,
      measurements_trigger=None,
  )


def create_task_layer(
    *,
    commands_processors: (
        Sequence[reaf_commands_processor.CommandsProcessor] | None
    ) = None,
    features_producers: (
        Sequence[reaf_features_producer.FeaturesProducer] | None
    ) = None,
    reward_provider: reaf_reward_provider.RewardProvider | None = None,
    termination_checkers: (
        Sequence[reaf_termination_checker.TerminationChecker] | None
    ) = None,
    discount_provider: reaf_discount_provider.DiscountProvider | None = None,
    features_observers: (
        Sequence[reaf_features_observers.FeaturesObserver] | None
    ) = None,
    loggers: Sequence[reaf_logger.Logger] | None = None,
) -> reaf_tll.TaskLogicLayer:
  """Returns a TaskLogicLayer.

  Args:
    commands_processors: commands processors, defaults to an empty list.
    features_producers:  features producers, defaults to an empty list.
    reward_provider: A reward provider to be used, if none is given a
      FakeRewardProvider is used.
    termination_checkers: termination checkers, if none is given a
      FakeTerminationChecker is used.
    discount_provider: A discount provider, if none is given a
      DefaultDiscountProvider is used.
    features_observers: features observers, defaults to an empty list.
    loggers: loggers, defaults to an empty list.
  """
  return reaf_tll.TaskLogicLayer(
      commands_processors=commands_processors or [],
      features_producers=features_producers or [],
      reward_provider=reward_provider or RandomRewardProvider(),
      termination_checkers=termination_checkers or [FakeTerminationChecker()],
      discount_provider=discount_provider
      or default_discount_provider.DefaultDiscountProvider(),
      features_observers=features_observers or [],
      loggers=loggers or [],
  )


def prefixing_action_adapter(
    command_prefix: str,
    action_spec: specs.BoundedArray,
) -> action_space_adapter.ActionSpaceAdapter:
  return partitioner_action_space_adapter.PartitionerActionSpaceAdapter(
      partitions=[
          partitioner_action_space_adapter.PartitionInfo(
              command_key=command_prefix
          )
      ],
      action_spec=action_spec,
  )


def no_op_observation_adapter(
    features_spec: Mapping[str, specs.Array],
) -> observation_space_adapter.ObservationSpaceAdapter:
  return default_observation_space_adapter.DefaultObservationSpaceAdapter(
      task_features_spec=features_spec,
      selected_features=None,
      renamed_features=None,
      observation_type_mapper=None,
  )


class FakeDevice(device.Device):
  """A device which records commands received and measurements returned."""

  @classmethod
  def one_command_one_measurement(
      cls,
      name: str,
      command_name: str,
      command_shape: tuple[int, ...],
      measurement_name: str,
      measurement_shape: tuple[int, ...],
  ) -> "FakeDevice":
    return cls(
        name=name,
        commands_spec={
            command_name: specs.BoundedArray(
                shape=command_shape,
                dtype=np.float32,
                minimum=np.finfo(np.float32).min,
                maximum=np.finfo(np.float32).max,
            )
        },
        measurements_spec={
            measurement_name: specs.Array(
                shape=measurement_shape, dtype=np.float32
            )
        },
    )

  def __init__(
      self,
      name: str = "fake_device",
      *,
      commands_spec: Mapping[str, gdmr_types.AnyArraySpec] | None = None,
      measurements_spec: Mapping[str, specs.Array] | None = None,
  ):
    if commands_spec is None:
      commands_spec = {
          f"{name}_{test_specs.random_string()}": test_specs.random_array_spec()
      }
    if measurements_spec is None:
      measurements_spec = {
          f"{name}_{test_specs.random_string()}": test_specs.random_array_spec()
      }
    self._name = name
    self._commands_spec = commands_spec
    self._measuremets_spec = measurements_spec
    self.recorded_commands = []
    self.returned_measurements = []

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return self._commands_spec

  @override
  def measurements_spec(self) -> Mapping[str, specs.Array]:
    return self._measuremets_spec

  @override
  def set_commands(self, commands: Mapping[str, gdmr_types.ArrayType]) -> None:
    self.recorded_commands.append(commands)

  @override
  def get_measurements(self) -> Mapping[str, gdmr_types.ArrayType]:
    measurements = test_specs.valid_dict_value(self._measuremets_spec)
    self.returned_measurements.append(measurements)
    return measurements


class FakeDeviceCoordinator(device_coordinator.DeviceCoordinator):
  """A fake (no-op) device coordinator."""

  def __init__(
      self,
      name: str = "fake_device_coordinator",
      *,
      devices: list[device.Device],
  ):
    self._name = name
    self._devices = devices

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def get_devices(self) -> list[device.Device]:
    return self._devices


class RandomRewardProvider(reaf_reward_provider.RewardProvider):
  """A fake (no-op) reward provider."""

  def __init__(
      self,
      name: str = "fake_reward_provider",
  ):
    self._name = name

  @override
  def name(self) -> str:
    return self._name

  @override
  def reward_spec(self) -> tree.Structure[specs.Array]:
    return specs.Array(shape=(1,), dtype=np.float32)

  @override
  def compute_reward(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> tree.Structure[gdmr_types.ArrayType]:
    return np.random.random(size=(1,)).astype(np.float32)

  @override
  def required_features_keys(self) -> set[str]:
    return set()


class FakeTerminationChecker(reaf_termination_checker.TerminationChecker):
  """A fake termination with a configurable step range."""

  def __init__(
      self,
      name: str = "fake_termination_checker",
      *,
      step_range: tuple[int, int] | None = None,
  ):
    if step_range is None:
      min_steps = random.randint(0, 10)
      max_steps = min_steps + random.randint(1, 10)
      step_range = (min_steps, max_steps)

    self._name = name
    self._step_range = step_range
    self._step_count = 0

  @override
  def name(self) -> str:
    return self._name

  @override
  def check_termination(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> reaf_termination_checker.TerminationResult:
    self._step_count += 1

    before_range = self._step_count < self._step_range[0]
    in_range = (
        self._step_count >= self._step_range[0]
        and self._step_count < self._step_range[1]
    )

    if before_range:
      return reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE
    elif in_range:
      if random.random() < 0.5:
        return reaf_termination_checker.TerminationResult.TERMINATE
      else:
        return reaf_termination_checker.TerminationResult.DO_NOT_TERMINATE
    else:
      return reaf_termination_checker.TerminationResult.TERMINATE

  @override
  def required_features_keys(self) -> set[str]:
    return set()


def no_op_reset() -> environment.EnvironmentReset:
  return environment_reset_from_callable.EnvironmentResetFromCallable(
      lambda unused_options: None
  )


class CommandRenamer(reaf_commands_processor.CommandsProcessor):
  """Renames commands."""

  def __init__(
      self,
      low_to_high_level_mapping: dict[str, str],
      dacl_spec: Mapping[str, gdmr_types.AnyArraySpec],
  ):
    self._consumed_commands_spec = {
        high_level_name: dacl_spec[low_level_name]
        for low_level_name, high_level_name in low_to_high_level_mapping.items()
    }
    self._low_to_high_level_mapping = low_to_high_level_mapping

  @override
  @property
  def name(self) -> str:
    return "command_renamer"

  @override
  def process_commands(
      self, consumed_commands: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:

    return {
        low: consumed_commands[high]
        for low, high in self._low_to_high_level_mapping.items()
    }

  @override
  def consumed_commands_spec(self) -> Mapping[str, gdmr_types.AnyArraySpec]:
    return self._consumed_commands_spec

  @override
  def produced_commands_keys(self) -> set[str]:
    return set(self._low_to_high_level_mapping.keys())
