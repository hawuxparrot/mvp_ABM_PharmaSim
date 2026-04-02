add_library(engine_core STATIC
  engine_input_validate.cpp
  simulation_state.cpp
  simulator.cpp
)

target_include_directories(engine_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(engine_core PUBLIC cxx_std_20)
