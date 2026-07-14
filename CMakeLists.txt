# Copyright 2026 Google LLC
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
cmake_minimum_required(VERSION 3.24)
project(reaf LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_FINDPYTHON ON)  # Forces Pybind11 to search for the interpreter.
# Explicitly find Python. This is to avoid issues when PyBind tries to find it
# (incorrectly).
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

option(REAF_USE_SYSTEM_ABSEIL "Whether to use system Abseil" OFF)

if(REAF_USE_SYSTEM_ABSEIL)
    set(ABSEIL_FIND_ARGS FIND_PACKAGE_ARGS NAMES absl)
else()
    set(ABSEIL_FIND_ARGS "")
endif()

include(FetchContent)

FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20260526.0
    ${ABSEIL_FIND_ARGS}
    )
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

set(USE_SYSTEM_PYBIND ON)  # Disable fetch pybind11 from pybind11-abseil.
FetchContent_Declare(
  pybind11_abseil
  GIT_REPOSITORY https://github.com/pybind/pybind11_abseil.git
  GIT_TAG fbc546beb4f798f56df2a44810e45327f36c1e52  # 03/04/2026
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
target_link_libraries(reaf_clock PUBLIC absl::time absl::log absl::clock_interface)

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
