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
"""Helpers for nesting and flattening dictionaries and specs."""

from collections.abc import Mapping
from typing import Any
from dm_env import specs
from gdm_robotics.interfaces import types as gdmr_types
import immutabledict


def flatten_spec(
    nested_dict: Mapping[str, Any], parent_key: str = '', separator: str = '.'
) -> Mapping[str, gdmr_types.AnyArraySpec]:
  """Flattens a spec, concatenating keys with the given separator."""
  return flatten_dict(nested_dict, parent_key, separator)


def nest_spec(
    flat_dict: Mapping[str, specs.Array], separator: str = '.'
) -> dict[Any, Any]:
  """Nests a spec, nesting by the given separator."""
  return nest_dict(flat_dict, separator)


def flatten_dict(
    nested_dict: Mapping[str, Any], parent_key: str = '', separator: str = '.'
) -> dict[Any, Any]:
  """Flattens a dict, concatenating keys with the given separator."""
  flat_items = []
  for k, v in nested_dict.items():
    next_key = parent_key + separator + k if parent_key else k
    if isinstance(v, dict) or isinstance(v, immutabledict.immutabledict):
      flat_items.extend(flatten_dict(v, next_key, separator).items())
    else:
      flat_items.append((next_key, v))
  return dict(flat_items)


def nest_dict(
    flat_dict: Mapping[str, Any], separator: str = '.'
) -> dict[Any, Any]:
  """Nests a dict, nesting by the given separator."""
  nested_dict = {}
  for k, v in flat_dict.items():
    nested_keys = k.split(separator)
    current_dict = nested_dict

    # For all non-terminal keys, nest the dictionary.
    for current_key in nested_keys[:-1]:
      if current_key in current_dict:
        current_dict = current_dict[current_key]
      else:
        current_dict[current_key] = {}
        current_dict = current_dict[current_key]

    # Add the entry to the dict for the terminal key.
    current_dict[nested_keys[-1]] = v
  return nested_dict
