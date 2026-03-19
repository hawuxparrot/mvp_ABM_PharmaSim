cmake_minimum_required(VERSION 3.24)
project(pharmasim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PHARMASIM_BUILD_PYTHON "Build nanobind module" ON)

add_subdirectory(cpp/engine)

if(PHARMASIM_BUILD_PYTHON)
  add_subdirectory(cpp/bindings)
endif()