find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_pharmasim_native
  dummy.cpp
)

target_link_libraries(_pharmasim_native PRIVATE engine_core)

# Install extension into Python package path (runtime package under ./python/runtime)
install(TARGETS _pharmasim_native LIBRARY DESTINATION runtime)