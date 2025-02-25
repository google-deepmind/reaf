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
"""APIs to connect to the fake robot used in the codelab."""

import threading
import time
import numpy as np

ROBOT_MOTOR_CONSTANT = 0.24  # Nm/A


_DT = 0.01


class Robot:
  """Simulates a real robot as a double integrator."""

  def __init__(self, num_dofs: int):
    self._dofs = num_dofs
    self._measurements_mutex = threading.Lock()
    # Protected by self._measurements_mutex.
    self._position = np.zeros(num_dofs)
    self._velocity = np.zeros(num_dofs)

    self._command_mutex = threading.Lock()
    # Protected by self._command_mutex.
    self._current_reference = np.zeros(num_dofs)

    self._should_stop = False
    self._thread = threading.Thread(target=self._integrator)
    self._thread.start()

  @property
  def dofs(self) -> int:
    return self._dofs

  def shutdown(self) -> None:
    self._should_stop = True

  def reset_state(
      self, new_position: np.ndarray, new_velocity: np.ndarray | None = None
  ) -> None:
    with self._command_mutex:
      self._current_reference = np.zeros(self._dofs)
      # Keep the lock.
      with self._measurements_mutex:
        self._position = new_position
        if new_velocity is not None:
          self._velocity = new_velocity
        else:
          self._velocity = np.zeros(self._dofs)

  def set_currents(self, currents: np.ndarray) -> None:
    with self._command_mutex:
      self._current_reference = currents

  def get_state(self) -> tuple[np.ndarray, np.ndarray]:
    with self._measurements_mutex:
      return self._position, self._velocity

  def _integrator(self) -> None:
    """Integrates the system dynamics."""
    # Initialize time.
    last_sleep_time = time.time()
    while not self._should_stop:
      # Get currents.
      with self._command_mutex:
        currents = self._current_reference

      with self._measurements_mutex:
        self._position += _DT * self._velocity
        self._velocity += _DT * currents

      last_sleep_time += _DT
      time.sleep(last_sleep_time - time.time())
