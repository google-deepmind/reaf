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
import operator
from typing import Callable, Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from reaf.core import reward_provider
import tree

from absl.testing import absltest
from absl.testing import parameterized


class TestRewardProvider(reward_provider.RewardProvider):

  def __init__(
      self,
      required_features_spec: Mapping[str, specs.Array],
      reward_callable: Callable[
          [Mapping[str, gdmr_types.ArrayType]], reward_provider.RewardValue
      ],
  ):
    super().__init__()
    self._required_features_spec = required_features_spec
    self._reward_callable = reward_callable

  def name(self) -> str:
    return 'test'

  def compute_reward(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> reward_provider.RewardValue:
    return self._reward_callable(required_features)

  def reward_spec(self) -> reward_provider.RewardSpec:
    dummy_features = tree.map_structure(
        lambda s: s.generate_value(), self._required_features_spec
    )
    dummy_reward = self._reward_callable(dummy_features)
    return tree.map_structure(
        lambda v: specs.Array(v.shape, v.dtype), dummy_reward
    )

  def required_features_keys(self) -> set[str]:
    return set(self._required_features_spec.keys())


@parameterized.named_parameters(*[
    dict(
        testcase_name=op.__name__,
        op=op,
    )
    for op in (
        operator.add,
        operator.mul,
        operator.sub,
        operator.truediv,
        operator.floordiv,
        operator.pow,
    )
])
class JAXRewardProviderTest(parameterized.TestCase):

  def test_binary_operator_scalar(self, op: reward_provider.BinaryOperator):
    first_provider = reward_provider.ConstantRewardProvider(
        reward=np.array(1.0, dtype=np.float32)
    )
    second_provider = reward_provider.ConstantRewardProvider(
        reward=np.array(2.0, dtype=np.float32)
    )
    provider = op(first_provider, second_provider)
    required_features = {}
    self.assertEqual(
        provider.required_features_keys(), required_features.keys()
    )
    self.assertEqual(
        provider.reward_spec(), specs.Array(shape=(), dtype=np.float32)
    )
    expected_reward = op(1.0, 2.0)
    self.assertEqual(
        provider.compute_reward(required_features), expected_reward
    )

  def test_binary_operator_primitive(self, op: reward_provider.BinaryOperator):
    first_provider = reward_provider.ConstantRewardProvider(
        reward=np.array(1.0, dtype=np.float32)
    )
    second_provider = np.array(2.0, dtype=np.float32)
    provider = op(first_provider, second_provider)
    required_features = {}
    self.assertEqual(
        provider.required_features_keys(), required_features.keys()
    )
    self.assertEqual(
        provider.reward_spec(), specs.Array(shape=(), dtype=np.float32)
    )
    expected_reward = op(1.0, 2.0)
    self.assertEqual(
        provider.compute_reward(required_features), expected_reward
    )

  def test_binary_operator_same_structure(
      self, op: reward_provider.BinaryOperator
  ):
    first_provider = reward_provider.ConstantRewardProvider(
        reward=[
            np.array(1.0, dtype=np.float32),
            {
                'foo': np.array(2.0, dtype=np.float32),
                'bar': np.array(3.0, dtype=np.float32),
            },
        ]
    )
    second_provider = reward_provider.ConstantRewardProvider(
        reward=[
            np.array(4.0, dtype=np.float32),
            {
                'foo': np.array(5.0, dtype=np.float32),
                'bar': np.array(6.0, dtype=np.float32),
            },
        ]
    )
    provider = op(first_provider, second_provider)
    required_features = {}
    self.assertEqual(
        provider.required_features_keys(), required_features.keys()
    )
    self.assertEqual(
        provider.reward_spec(),
        [
            specs.Array(shape=(), dtype=np.float32),
            {
                'foo': specs.Array(shape=(), dtype=np.float32),
                'bar': specs.Array(shape=(), dtype=np.float32),
            },
        ],
    )
    expected_reward = [op(1.0, 4.0), {'foo': op(2.0, 5.0), 'bar': op(3.0, 6.0)}]
    self.assertEqual(
        provider.compute_reward(required_features), expected_reward
    )

  def test_binary_operator_different_structure(
      self, op: reward_provider.BinaryOperator
  ):
    first_provider = reward_provider.ConstantRewardProvider(
        reward=[
            np.array(1.0, dtype=np.float32),
            {
                'foo': np.array(2.0, dtype=np.float32),
                'bar': np.array(3.0, dtype=np.float32),
            },
        ]
    )
    second_provider = reward_provider.ConstantRewardProvider(
        reward=[
            np.array(4.0, dtype=np.float32),
            {
                'foo': np.array(5.0, dtype=np.float32),
            },
        ]
    )
    with self.assertRaises(ValueError):
      _ = op(first_provider, second_provider)

  def test_binary_operator_from_features(
      self, op: reward_provider.BinaryOperator
  ):
    first_provider = TestRewardProvider(
        required_features_spec={'foo': specs.Array(shape=(), dtype=np.float32)},
        reward_callable=lambda required_features: required_features['foo'],
    )
    second_provider = TestRewardProvider(
        required_features_spec={'bar': specs.Array(shape=(), dtype=np.float32)},
        reward_callable=lambda required_features: required_features['bar'],
    )
    provider = op(first_provider, second_provider)
    required_features = {
        'foo': np.array(1.0, dtype=np.float32),
        'bar': np.array(2.0, dtype=np.float32),
    }
    self.assertEqual(
        provider.required_features_keys(), required_features.keys()
    )
    self.assertEqual(
        provider.reward_spec(), specs.Array(shape=(), dtype=np.float32)
    )
    expected_reward = op(1.0, 2.0)
    self.assertEqual(
        provider.compute_reward(required_features), expected_reward
    )


if __name__ == '__main__':
  absltest.main()
