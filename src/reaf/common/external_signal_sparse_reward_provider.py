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
"""Reward provider that returns a sparse reward based on an external signal."""

from collections.abc import Mapping
import threading

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import reward_provider
import tree
from typing_extensions import override


class ExternalSignalSparseRewardProvider(reward_provider.RewardProvider):
  """Sets a reward of `1.0` based on an external signal, or `0.0` otherwise.

  This reward provider allows external users to control the sparse reward of
  an episode.

  On construction, this sparse provider will always return `0.0` as the reward.
  The user can call `set_success()` to signal that the conditions for a
  successful reward have been met. Every subsequent call to `compute_reward()`
  will return `1.0` until the provider is reset by calling `clear_success()`,
  e.g. in an episode reset function.
  """

  def __init__(self, name: str):
    """Initializes the reward provider."""
    self._name = name
    self._event = threading.Event()

  def set_success(self) -> None:
    """Signal that the provider should return `1.0` on subsequent calls."""
    self._event.set()

  @override
  def name(self) -> str:
    """Returns a unique string identifier for this object."""
    return self._name

  @override
  def compute_reward(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> tree.Structure[gdmr_types.ArrayType]:
    """Returns the reward."""
    del required_features

    if self._event.is_set():
      return np.ones(1)

    return np.zeros(1)

  @override
  def reward_spec(self) -> tree.Structure[specs.Array]:
    """Returns the spec for a constant zero reward."""
    return specs.Array(shape=(1,), dtype=float)

  @override
  def required_features_keys(self) -> set[str]:
    """Returns an empty set.

    There are no feature keys that are required to compute the reward.
    """
    return set()

  @override
  def reset(self) -> None:
    self._event.clear()
