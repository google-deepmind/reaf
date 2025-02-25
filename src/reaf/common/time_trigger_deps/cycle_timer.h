// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef REAF_COMMON_TIME_TRIGGER_DEPS_CYCLE_TIMER_H_
#define REAF_COMMON_TIME_TRIGGER_DEPS_CYCLE_TIMER_H_

#include <memory>

#include <absl/time/time.h>
#include "clock.h"

namespace reaf {

class CycleTimer {
 public:
  CycleTimer(std::unique_ptr<reaf::Clock> clock, absl::Duration period);

  void WaitForNextPeriod();

 private:
  std::unique_ptr<reaf::Clock> clock_;
  absl::Duration period_;
  absl::Time last_run_;
};

}  // namespace reaf

#endif  // REAF_COMMON_TIME_TRIGGER_DEPS_CYCLE_TIMER_H_
