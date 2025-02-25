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
import numpy as np
from reaf.common import external_signal_sparse_reward_provider
from reaf.testing import tree_spec_validation_assertions
from absl.testing import absltest


class ExternalSignalSparseRewardProviderTest(
    tree_spec_validation_assertions.TreeSpecValidationAssertions,
    absltest.TestCase,
):

  def test_name(self):
    reward_provider = external_signal_sparse_reward_provider.ExternalSignalSparseRewardProvider(
        name="test_name"
    )
    self.assertEqual(reward_provider.name(), "test_name")

  def test_compute_reward(self):
    # Zero on construction.
    reward_provider = external_signal_sparse_reward_provider.ExternalSignalSparseRewardProvider(
        "test"
    )
    self.assertEqual(reward_provider.compute_reward({}), np.zeros(1))

    # One after signalling.
    reward_provider.set_success()
    self.assertEqual(reward_provider.compute_reward({}), np.ones(1))

    # Zero after resetting.
    reward_provider.reset()
    self.assertEqual(reward_provider.compute_reward({}), np.zeros(1))

  def test_reward_spec(self):
    reward_provider = external_signal_sparse_reward_provider.ExternalSignalSparseRewardProvider(
        "test"
    )
    self.assert_tree_matches_spec(
        tree_structure=reward_provider.compute_reward({}),
        spec=reward_provider.reward_spec(),
    )

  def test_required_features_keys(self):
    reward_provider = external_signal_sparse_reward_provider.ExternalSignalSparseRewardProvider(
        "test"
    )
    self.assertEqual(reward_provider.required_features_keys(), set())


if __name__ == "__main__":
  absltest.main()
