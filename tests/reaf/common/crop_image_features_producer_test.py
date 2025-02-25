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
from dm_env import specs
import numpy as np
from reaf.common import crop_image_features_producer
from absl.testing import absltest
from absl.testing import parameterized


class CropImageFeaturesProducerTest(parameterized.TestCase):

  def test_name(self):
    producer = crop_image_features_producer.CropImageFeaturesProducer(
        name='test_producer',
        image_feature_key='image',
        image_feature_spec=specs.Array(shape=(10, 15, 3), dtype=np.uint8),
        cropped_width=5,
        cropped_height=4,
        upper_left_corner_row=2,
        upper_left_corner_column=3,
    )
    self.assertEqual(producer.name, 'test_producer')

  def test_required_features_keys(self):
    producer = crop_image_features_producer.CropImageFeaturesProducer(
        name='test_producer',
        image_feature_key='image',
        image_feature_spec=specs.Array(shape=(10, 15, 3), dtype=np.uint8),
        cropped_width=5,
        cropped_height=4,
        upper_left_corner_row=2,
        upper_left_corner_column=3,
    )
    self.assertSetEqual(producer.required_features_keys(), {'image'})

  @parameterized.named_parameters(
      dict(
          testcase_name='valid_crop',
          image_feature_key='image',
          image_shape=(10, 15, 3),
          cropped_width=5,
          cropped_height=4,
          upper_left_corner_row=2,
          upper_left_corner_column=3,
          expected_cropped_shape=(4, 5, 3),
      ),
      dict(
          testcase_name='full_image_crop',
          image_feature_key='rgb_image',
          image_shape=(10, 15, 3),
          cropped_width=15,
          cropped_height=10,
          upper_left_corner_row=0,
          upper_left_corner_column=0,
          expected_cropped_shape=(10, 15, 3),
      ),
  )
  def test_crop_image_features_producer(
      self,
      image_feature_key: str,
      image_shape: tuple[int, int, int],
      cropped_width: int,
      cropped_height: int,
      upper_left_corner_row: int,
      upper_left_corner_column: int,
      expected_cropped_shape: tuple[int, int, int],
  ):
    producer = crop_image_features_producer.CropImageFeaturesProducer(
        name='test_producer',
        image_feature_key=image_feature_key,
        image_feature_spec=specs.Array(shape=image_shape, dtype=np.uint8),
        cropped_width=cropped_width,
        cropped_height=cropped_height,
        upper_left_corner_row=upper_left_corner_row,
        upper_left_corner_column=upper_left_corner_column,
    )

    produced_features_spec = producer.produced_features_spec()
    self.assertLen(produced_features_spec, 1)
    self.assertIn(
        f'{image_feature_key}_cropped_{cropped_width}x{cropped_height}',
        produced_features_spec,
    )
    self.assertEqual(
        produced_features_spec[
            f'{image_feature_key}_cropped_{cropped_width}x{cropped_height}'
        ].shape,
        expected_cropped_shape,
    )

    image = (
        np.arange(1, np.prod(image_shape) + 1)
        .reshape(image_shape)
        .astype(np.uint8)
    )
    cropped_image = producer.produce_features({image_feature_key: image})[
        f'{image_feature_key}_cropped_{cropped_width}x{cropped_height}'
    ]
    self.assertEqual(cropped_image.shape, expected_cropped_shape)
    np.testing.assert_array_equal(
        cropped_image,
        image[
            upper_left_corner_row : upper_left_corner_row + cropped_height,
            upper_left_corner_column : upper_left_corner_column + cropped_width,
        ],
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='negative_upper_left_corner_row',
          upper_left_corner_row=-1,
          upper_left_corner_column=3,
          cropped_width=5,
          cropped_height=4,
      ),
      dict(
          testcase_name='negative_upper_left_corner_column',
          upper_left_corner_row=2,
          upper_left_corner_column=-3,
          cropped_width=5,
          cropped_height=4,
      ),
      dict(
          testcase_name='negative_cropped_width',
          upper_left_corner_row=2,
          upper_left_corner_column=3,
          cropped_width=-5,
          cropped_height=4,
      ),
      dict(
          testcase_name='negative_cropped_height',
          upper_left_corner_row=2,
          upper_left_corner_column=3,
          cropped_width=5,
          cropped_height=-4,
      ),
  )
  def test_negative_input_raises_value_error(
      self,
      upper_left_corner_row: int,
      upper_left_corner_column: int,
      cropped_width: int,
      cropped_height: int,
  ):
    with self.assertRaisesRegex(ValueError, 'must be non-negative'):
      crop_image_features_producer.CropImageFeaturesProducer(
          name='test_producer',
          image_feature_key='image',
          image_feature_spec=specs.Array(shape=(10, 15, 3), dtype=np.uint8),
          cropped_width=cropped_width,
          cropped_height=cropped_height,
          upper_left_corner_row=upper_left_corner_row,
          upper_left_corner_column=upper_left_corner_column,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='upper_left_corner_row_beyond_image_height',
          upper_left_corner_row=12,
          upper_left_corner_column=3,
          cropped_width=5,
          cropped_height=4,
      ),
      dict(
          testcase_name='upper_left_corner_column_beyond_image_width',
          upper_left_corner_row=2,
          upper_left_corner_column=35,
          cropped_width=5,
          cropped_height=4,
      ),
      dict(
          testcase_name='cropped_width_beyond_image_width',
          upper_left_corner_row=2,
          upper_left_corner_column=8,
          cropped_width=13,
          cropped_height=4,
      ),
      dict(
          testcase_name='cropped_height_beyond_image_height',
          upper_left_corner_row=7,
          upper_left_corner_column=3,
          cropped_width=5,
          cropped_height=10,
      ),
  )
  def test_raises_when_cropped_area_exceeds_image_bounds(
      self,
      upper_left_corner_row: int,
      upper_left_corner_column: int,
      cropped_width: int,
      cropped_height: int,
  ):
    with self.assertRaisesRegex(ValueError, 'exceeds the bounds'):
      crop_image_features_producer.CropImageFeaturesProducer(
          name='test_producer',
          image_feature_key='image',
          image_feature_spec=specs.Array(shape=(10, 15, 3), dtype=np.uint8),
          cropped_width=cropped_width,
          cropped_height=cropped_height,
          upper_left_corner_row=upper_left_corner_row,
          upper_left_corner_column=upper_left_corner_column,
      )

  def test_wrong_image_spec_shape_raises_value_error(self):
    with self.assertRaisesRegex(ValueError, 'must have 3 dimensions'):
      crop_image_features_producer.CropImageFeaturesProducer(
          name='test_producer',
          image_feature_key='image',
          image_feature_spec=specs.Array(shape=(10, 15), dtype=np.uint8),
          cropped_width=5,
          cropped_height=4,
          upper_left_corner_row=2,
          upper_left_corner_column=3,
      )

  def test_new_feature_name(self):
    image_shape = (10, 15, 3)
    image_feature_key = 'image'
    new_feature_name = 'cropped_image'
    producer = crop_image_features_producer.CropImageFeaturesProducer(
        name='test_producer',
        image_feature_key=image_feature_key,
        image_feature_spec=specs.Array(shape=image_shape, dtype=np.uint8),
        cropped_width=5,
        cropped_height=4,
        upper_left_corner_row=2,
        upper_left_corner_column=3,
        new_feature_name=new_feature_name,
    )
    produced_features_spec = producer.produced_features_spec()
    self.assertLen(produced_features_spec, 1)
    self.assertIn(new_feature_name, produced_features_spec)

    self.assertIn(
        new_feature_name,
        producer.produce_features({
            image_feature_key: (
                np.arange(1, np.prod(image_shape) + 1)
                .reshape(image_shape)
                .astype(np.uint8)
            )
        }),
    )


if __name__ == '__main__':
  absltest.main()
