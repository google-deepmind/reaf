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

cmake_minimum_required(VERSION 3.15)
project(reaf LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
set(USE_SYSTEM_PYBIND ON)  # Disable fetch pybind11 from pybind11-abseil.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Explicitly find Python. This is to avoid issues when PyBind tries to find it
# (incorrectly).
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)


include(FetchContent)
FetchContent_Declare(
  abseil-cpp
  # Specify the commit you depend on and update it regularly.
  URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.tar.gz
)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

# Note: pybind11_abseil will internally try to fetch both abseil-cpp and
# pybind11.
# abseil-cpp is not a problem as it is declared above and our declaration, being
# done first, takes precedence.
# pybind11 is more of a problem. We will allow pybind11_abseil to fetch it and
# we will check for its existence just to be on the safe side.
FetchContent_Declare(
    pybind11_abseil
    GIT_REPOSITORY https://github.com/pybind/pybind11_abseil.git
    GIT_TAG master
  )
# pybind11_abseil internally forces interprocedural optimization to off. But
# this is not propagated up to the top level CMakeLists.txt, which means that
# there will be incompatibility between our targets and the one defined by
# pybind11_abseil.
# We force it to on here.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(abseil-cpp pybind11_abseil)

# Cycle timer dependencies
add_library(reaf_clock)
target_sources(reaf_clock PRIVATE src/reaf/common/time_trigger_deps/clock.h
                                  src/reaf/common/time_trigger_deps/clock.cc)
target_include_directories(reaf_clock PRIVATE src/reaf/common/time_trigger_deps)
target_link_libraries(reaf_clock PUBLIC absl::time absl::log)

add_library(cycle_timer_cc)
target_sources(cycle_timer_cc PRIVATE src/reaf/common/time_trigger_deps/cycle_timer.h
                                      src/reaf/common/time_trigger_deps/cycle_timer.cc)
target_include_directories(cycle_timer_cc PRIVATE src/reaf/common/time_trigger_deps)
target_link_libraries(cycle_timer_cc PUBLIC reaf_clock absl::time absl::log absl::check absl::str_format)

# Cycle timer module.
pybind11_add_module(cycle_timer src/reaf/common/time_trigger_deps/python/cycle_timer.cc)
target_include_directories(cycle_timer PRIVATE src/reaf/common/time_trigger_deps)
target_link_libraries(cycle_timer PUBLIC
    cycle_timer_cc reaf_clock absl::time absl::log absl::check absl::str_format
    absl::memory pybind11_abseil::absl_casters)

# Destination is in the target Python path, i.e. without src.
install(TARGETS cycle_timer DESTINATION reaf/common/time_trigger_deps/python)
