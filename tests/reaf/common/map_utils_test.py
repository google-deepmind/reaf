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
"""Tests for map utils."""

from absl import logging
from dm_env import specs
import numpy as np
from reaf.common import map_utils
from absl.testing import absltest


class MapUtilsTest(absltest.TestCase):

  def test_nest_flaten_dict(self):
    nested_dict = {'a': {'b': 1}}
    flat_dict = map_utils.flatten_dict(nested_dict)
    logging.info('flat_dict: %s', flat_dict)
    self.assertEqual(flat_dict['a.b'], 1)
    re_nested_dict = map_utils.nest_dict(flat_dict)
    logging.info('re_nested_dict: %s', re_nested_dict)
    self.assertEqual(re_nested_dict, nested_dict)

  def test_nest_flatten_spec(self):
    array_spec = specs.Array(shape=(1, 2), dtype=np.int32)
    nested_spec = {'a': {'b': array_spec}}
    flat_spec = map_utils.flatten_spec(nested_spec)
    logging.info('flat_spec: %s', flat_spec)
    self.assertEqual(flat_spec['a.b'], array_spec)
    re_nested_spec = map_utils.nest_spec(flat_spec)
    logging.info('re_nested_dict: %s', re_nested_spec)
    self.assertEqual(re_nested_spec, nested_spec)


if __name__ == '__main__':
  absltest.main()
