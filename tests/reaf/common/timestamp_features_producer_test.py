import time

from dm_env import specs
import numpy as np
from reaf.common import timestamp_features_producer

from absl.testing import absltest
from absl.testing import parameterized


class TimestampFeaturesProducerTest(parameterized.TestCase):

  def test_produces_timestamp_feature(self):
    producer = timestamp_features_producer.TimestampFeaturesProducer(
        timestamp_key="timestamp"
    )
    features = producer.produce_features({})
    self.assertIn("timestamp", features)
    self.assertIsInstance(features["timestamp"], np.ndarray)
    self.assertEqual(features["timestamp"].dtype, np.int64)
    self.assertAlmostEqual(features["timestamp"], time.time_ns(), delta=1e8)

  def test_produced_features_spec(self):
    producer = timestamp_features_producer.TimestampFeaturesProducer(
        timestamp_key="timestamp",
    )
    spec = producer.produced_features_spec()
    self.assertIn("timestamp", spec)
    self.assertEqual(
        spec["timestamp"], specs.Array(shape=(), dtype=np.int64)
    )

  def test_required_features_keys(self):
    producer = timestamp_features_producer.TimestampFeaturesProducer(
        timestamp_key="timestamp"
    )
    self.assertEqual(producer.required_features_keys(), set())

  def test_name(self):
    producer = timestamp_features_producer.TimestampFeaturesProducer(
        timestamp_key="my_timestamp"
    )
    self.assertEqual(producer.name, "timestamp_features_producer_my_timestamp")


if __name__ == "__main__":
  absltest.main()
