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
"""Assertions for REAF devices."""

import unittest
import numpy as np


class DeviceAssertions(unittest.TestCase):
  """Mix this into a googletest.TestCase class to get assertions for devices.

  Usage:

    class SomeTestCase(
      device_assertions.DeviceAssertions,
      googletest.TestCase,
    ):
      ...
      def testSomething(self):
        ...
        self.assert_measurements_match_specs(device)
  """

  def assert_measurements_match_specs(self, device):
    measurements = device.get_measurements()
    specs = device.measurements_spec()

    self.assertEqual(measurements.keys(), specs.keys())
    for key, measurement in measurements.items():
      try:
        specs[key].validate(measurement)
      except Exception as e:  # pylint: disable=broad-exception-caught
        self.fail(f"Measurement spec mismatch at key {key}! Error: {e}.")

  def assert_measurements_almost_equal(
      self,
      actual: dict[str, np.ndarray],
      desired: dict[str, np.ndarray],
      decimal: int = 2,
  ):
    """Asserts measurements are equal up to desired precision.

    Args:
      actual: The actual measurements.
      desired: The desired measurements.
      decimal: The number of decimal places to match.
    """
    self.assertEqual(
        actual.keys(),
        desired.keys(),
        msg="Measurements dicts do not contain the same keys.",
    )
    for key in actual:
      np.testing.assert_almost_equal(actual[key], desired[key], decimal=decimal)
