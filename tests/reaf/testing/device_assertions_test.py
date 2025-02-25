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
"""Tests for device_assertions module."""

from dm_env import specs
import numpy as np
from reaf.core import device
from reaf.testing import device_assertions
from absl.testing import absltest
from absl.testing import parameterized


class TestDevice(device.Device):
  __test__ = False

  def __init__(self, measurements_spec, measurements):
    self._measurements_spec = measurements_spec
    self._measurements = measurements

  @property
  def name(self):
    return "test_device"

  def commands_spec(self):
    return {}

  def measurements_spec(self):
    return self._measurements_spec

  def set_commands(self, commands):
    pass

  def get_measurements(self):
    return self._measurements


class DeviceAssertionsTest(
    device_assertions.DeviceAssertions,
    parameterized.TestCase,
):

  def test_assert_measurements_match_specs_succeeds(self):
    measurements_spec = {"one": specs.Array(shape=(3, 4, 5), dtype=np.float64)}
    measurements = {"one": np.zeros(shape=(3, 4, 5), dtype=np.float64)}
    test_device = TestDevice(measurements_spec, measurements)
    self.assert_measurements_match_specs(test_device)

  @parameterized.named_parameters(
      dict(
          testcase_name="different_names",
          measurements_spec={"one": specs.Array(shape=(3,), dtype=np.float64)},
          measurements={"two": np.zeros(shape=(3,), dtype=np.float64)},
      ),
      dict(
          testcase_name="different_shapes",
          measurements_spec={
              "one": specs.Array(shape=(3, 3), dtype=np.float64)
          },
          measurements={"one": np.zeros(shape=(3,), dtype=np.float64)},
      ),
      dict(
          testcase_name="different_dtypes",
          measurements_spec={"one": specs.Array(shape=(3,), dtype=np.float32)},
          measurements={"one": np.zeros(shape=(3,), dtype=np.float64)},
      ),
      dict(
          testcase_name="missing_spec",
          measurements_spec={"one": specs.Array(shape=(3,), dtype=np.float64)},
          measurements={
              "one": np.zeros(shape=(3,), dtype=np.float64),
              "two": np.zeros(shape=(3,), dtype=np.float64),
          },
      ),
      dict(
          testcase_name="missing_measurement",
          measurements_spec={
              "one": specs.Array(shape=(3,), dtype=np.float64),
              "two": np.zeros(shape=(3,), dtype=np.float64),
          },
          measurements={
              "one": np.zeros(shape=(3,), dtype=np.float64),
          },
      ),
      dict(
          testcase_name="out_of_bounds",
          measurements_spec={
              "one": specs.BoundedArray(
                  (2,), np.float64, minimum=[-0.1, 0.2], maximum=[0.9, 0.9]
              ),
          },
          measurements={
              "one": np.zeros(shape=(2,), dtype=np.float64),
          },
      ),
  )
  def test_assert_measurements_match_specs_fails(
      self,
      measurements_spec,
      measurements,
  ):
    test_device = TestDevice(measurements_spec, measurements)
    with self.assertRaises(AssertionError):
      self.assert_measurements_match_specs(test_device)

  def test_assert_measurements_almost_equal_succeeds(self):
    actual_measurements = {"some_measurement": np.array([1.11, 2.22, 3.33])}
    expected_measurements = {"some_measurement": np.array([1.1, 2.2, 3.3])}
    self.assert_measurements_almost_equal(
        actual_measurements, expected_measurements, decimal=1
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="different_keys",
          actual_measurements={"some_measurement": np.array([1, 2, 3])},
          expected_measurements={"some_other_measurement": np.array([1, 2, 3])},
      ),
      dict(
          testcase_name="different_values",
          actual_measurements={"some_measurement": np.array([1, 2, 3])},
          expected_measurements={"some_measurement": np.array([1, 2])},
      ),
      dict(
          testcase_name="different_decimal_precision",
          actual_measurements={
              "some_measurement": np.array([1.11, 2.22, 3.33])
          },
          expected_measurements={"some_measurement": np.array([1.1, 2.2, 3.3])},
      ),
  )
  def test_assert_measurements_almost_equal_fails(
      self, actual_measurements, expected_measurements
  ):
    with self.assertRaises(AssertionError):
      self.assert_measurements_almost_equal(
          actual_measurements, expected_measurements, decimal=2
      )


if __name__ == "__main__":
  absltest.main()
