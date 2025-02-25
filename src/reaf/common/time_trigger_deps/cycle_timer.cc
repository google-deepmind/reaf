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
#include "cycle_timer.h"

#include <memory>
#include <utility>

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/strings/str_format.h>
#include <absl/time/time.h>
#include "clock.h"

namespace reaf {

CycleTimer::CycleTimer(std::unique_ptr<reaf::Clock> clock,
                       absl::Duration period)
    : clock_(std::move(clock)),
      period_(period),
      last_run_(absl::InfinitePast()) {
  // Should we move the check in an initialize method so that we can fail?
  CHECK_GT(period_, absl::ZeroDuration())
      << absl::StreamFormat("Period must be positive. Specified period: %s.",
                            absl::FormatDuration(period));
}

void CycleTimer::WaitForNextPeriod() {
  if (last_run_ == absl::InfinitePast()) {
    // This is the first time we run, so reset the last run time.
    last_run_ = clock_->TimeNow();
  }

  if (absl::Time now = clock_->TimeNow(); now >= last_run_ + period_) {
    LOG_EVERY_N_SEC(WARNING, 1) << absl::StreamFormat(
                                       "Previous cycle took too long. Expected "
                                       "%s. Got %s. This has happened ",
                                       absl::FormatDuration(period_),
                                       absl::FormatDuration(now - last_run_))
                                << COUNTER << " times so far.";
  }

  // Sleep until last_run_ + period_.
  clock_->SleepUntil(last_run_ + period_);

  // Note: we sleep by summing the constant period to the initial time, not by
  // using the current time. This should help preventing time drift.

  // If we get significantly behind (i.e. more than one cycle),
  // we catch up the start of the next period to the current time, to avoid
  // one lengthy cycle from causing a subsequent "pile-up" of zero-sleep
  // cycles.

  absl::Time next_period_time = last_run_ + period_;
  if (absl::Time now = clock_->TimeNow(); next_period_time + period_ < now) {
    // We are behind. Reset the clock.
    last_run_ = now;
  } else {
    // All good. Keep the constant period.
    last_run_ = next_period_time;
  }
}

}  // namespace reaf
