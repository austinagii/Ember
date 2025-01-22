cmake_minimum_required(VERSION 3.14)
project(ember)

set(CMAKE_CXX_STANDARD 20)

# Dependencies
include(FetchContent)
FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)

FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
  GIT_TAG        0.7.7
)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG       0.25.0
)

FetchContent_MakeAvailable(gtest xtl xtensor)

# Source files
set(EMBER_SOURCES
  src/ember/tensor.cpp
  src/ember/tensor_snapshot.cpp
  src/ember/autograd/accumulator.cpp
  src/ember/autograd/engine.cpp
  src/ember/autograd/edge.cpp
  src/ember/ops/add.cpp
  src/ember/ops/sub.cpp
  src/ember/ops/mul.cpp
  src/ember/ops/div.cpp
)

# Test files
set(EMBER_TESTS
  tests/ember/test_tensor.cpp
  tests/ember/ops/test_sub.cpp
  tests/ember/ops/test_add.cpp
  tests/ember/ops/test_mul.cpp
  tests/ember/ops/test_div.cpp
)

# Test executable
add_executable(ember_test ${EMBER_TESTS} ${EMBER_SOURCES})
target_include_directories(ember_test PRIVATE include)
target_link_libraries(ember_test PRIVATE 
    gtest_main 
    gmock_main
    xtl
    xtensor
)

add_test(NAME ember_test COMMAND ember_test)
