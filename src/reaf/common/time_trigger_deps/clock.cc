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
#include "clock.h"

#include <chrono>
#include <thread>

#include <absl/log/log.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

namespace reaf {
namespace {

class RealTimeClock : public Clock {
 public:
  ~RealTimeClock() override {
    LOG(FATAL) << "RealTimeClock should never be destroyed";
  }

  absl::Time TimeNow() override { return absl::Now(); }

  void SleepUntil(absl::Time wakeup_time) override {
    absl::Duration d = wakeup_time - TimeNow();
    if (d > absl::ZeroDuration()) {
      absl::SleepFor(d);
    }
  }
};

absl::Duration DurationSinceSteadyClockEpoch() {
  using clock = std::chrono::steady_clock;
  clock::time_point time = clock::now();
  clock::duration duration_since_epoch = time.time_since_epoch();
  return absl::FromChrono(std::chrono::duration_cast<std::chrono::nanoseconds>(
      duration_since_epoch));
}

}  // namespace

Clock::~Clock() {}
Clock *Clock::RealClock() {
  static RealTimeClock *clock = new RealTimeClock();
  return clock;
}

MonotonicClock::MonotonicClock()
    : steady_clock_epoch_(absl::Now() - DurationSinceSteadyClockEpoch()) {}
MonotonicClock::~MonotonicClock() {}

absl::Time MonotonicClock::TimeNow() {
  return steady_clock_epoch_ + DurationSinceSteadyClockEpoch();
}

void MonotonicClock::SleepUntil(absl::Time wakeup_time) {
  absl::Duration wakeup_time_since_epoch = wakeup_time - steady_clock_epoch_;
  std::chrono::steady_clock::time_point wakeup_time_in_steady_clock(
      absl::ToChronoNanoseconds(wakeup_time_since_epoch));

  std::this_thread::sleep_until(wakeup_time_in_steady_clock);
}

}  // namespace reaf
