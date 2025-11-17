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
"""A features producer that adds a timestamp to the features dictionary."""

from collections.abc import Mapping
import time
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import features_producer
from typing_extensions import override


class TimestampFeaturesProducer(features_producer.FeaturesProducer):
  """A features producer that adds a timestamp to the features dictionary."""

  def __init__(self, timestamp_key: str):
    """Initializes the timestamp features producer.

    Args:
      timestamp_key: The key to use for the timestamp feature.
    """
    self._timestamp_key = timestamp_key

  @property
  @override
  def name(self) -> str:
    return f"timestamp_features_producer_{self._timestamp_key}"

  @override
  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    return {self._timestamp_key: np.array(time.time_ns(), dtype=np.int64)}

  @override
  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return {self._timestamp_key: specs.Array(shape=(), dtype=np.int64)}

  @override
  def required_features_keys(self) -> set[str]:
    return set()
