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
"""Features producer that crops an image feature."""

from collections.abc import Mapping

from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
from reaf.core import features_producer
from typing_extensions import override


class CropImageFeaturesProducer(features_producer.FeaturesProducer):
  """Crops an image feature to the desired width and height."""

  def __init__(
      self,
      *,
      name: str,
      image_feature_key: str,
      image_feature_spec: specs.Array,
      cropped_width: int,
      cropped_height: int,
      upper_left_corner_row: int,
      upper_left_corner_column: int,
      new_feature_name: str | None = None,
  ):
    """Initializer.

    Args:
      name: Name of the features producer.
      image_feature_key: Key of the image feature to crop.
      image_feature_spec: Spec of the image feature to crop.
      cropped_width: Desired width of the cropped image.
      cropped_height: Desired height of the cropped image.
      upper_left_corner_row: Row index of the upper left corner of the crop
        area. This + the cropped height must be less than the image height.
      upper_left_corner_column: Column index of the upper left corner of the
        crop area. This + the cropped width must be less than the image width.
      new_feature_name: Name of the new feature. If None, the new feature will
        be named 'image_feature_key_cropped_{cropped_width}x{cropped_height}'.

    Raises:
      ValueError: If the cropped area exceeds the bounds of the input image.
      ValueError: If the image spec does not have 3 dimensions.
      ValueError: If the upper left corner coordinates are negative or
        the cropped dimensions are negative.
    """
    self._name = name
    self._image_feature_key = image_feature_key
    self._cropped_width = cropped_width
    self._cropped_height = cropped_height
    self._start_column = upper_left_corner_column
    self._end_column = self._start_column + self._cropped_width
    self._start_row = upper_left_corner_row
    self._end_row = self._start_row + self._cropped_height
    self._new_feature_name = new_feature_name or (
        f'{image_feature_key}_cropped_{cropped_width}x{cropped_height}'
    )

    if len(image_feature_spec.shape) != 3:
      raise ValueError(
          'The image spec must have 3 dimensions (height, width, channels). '
          f'Got {len(image_feature_spec.shape)} dimensions instead.'
      )
    height, width, channels = image_feature_spec.shape

    if (
        self._start_column < 0
        or self._start_row < 0
        or cropped_width <= 0
        or cropped_height <= 0
    ):
      raise ValueError(
          'The upper left corner coordinates and cropped dimensions must be'
          ' non-negative.'
      )
    if self._end_column > width or self._end_row > height:
      raise ValueError(
          'The desired crop area with upper left corner'
          f' ({upper_left_corner_row}, {upper_left_corner_column}), width'
          f' {cropped_width}, and height {cropped_height} exceeds the bounds of'
          f' the input image with shape {image_feature_spec.shape}.'
      )

    self._produce_features_spec = {
        self._new_feature_name: specs.Array(
            shape=(
                self._cropped_height,
                self._cropped_width,
                channels,
            ),
            dtype=image_feature_spec.dtype,
        )
    }

  @override
  @property
  def name(self) -> str:
    return self._name

  @override
  def produce_features(
      self, required_features: Mapping[str, gdmr_types.ArrayType]
  ) -> Mapping[str, gdmr_types.ArrayType]:
    """Returns the cropped the image feature."""
    return {
        self._new_feature_name: required_features[self._image_feature_key][
            self._start_row : self._end_row,
            self._start_column : self._end_column,
        ]
    }

  @override
  def produced_features_spec(self) -> Mapping[str, specs.Array]:
    return self._produce_features_spec

  @override
  def required_features_keys(self) -> set[str]:
    return {self._image_feature_key}
