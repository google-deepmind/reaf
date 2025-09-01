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
