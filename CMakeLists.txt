cmake_minimum_required(VERSION 3.14)
project(hypergrad)

set(CMAKE_CXX_STANDARD 20)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
# find_package(numpy REQUIRED)

include(FetchContent)

# Fetch xtl
FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git 
  GIT_TAG        0.7.7
)
FetchContent_MakeAvailable(xtl)

# Fetch xtensor
FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
  GIT_TAG        0.25.0
)
FetchContent_MakeAvailable(xtensor)

# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Fetch xtensor-python
FetchContent_Declare(
  xtensor-python
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-python.git
  GIT_TAG        0.27.0
)
FetchContent_MakeAvailable(xtensor-python)

FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
)
FetchContent_MakeAvailable(gtest)

set(SOURCES
  src/bindings.cpp
)

set(HEADERS 
  include/fcnn.h
)

pybind11_add_module(hyperpy ${SOURCES} ${HEADERS})

target_include_directories(hyperpy 
  PRIVATE 
    ${xtensor_SOURCE_DIR}/include 
    ${xtl_SOURCE_DIR}/include
    ${pybind11_SOURCE_DIR}/include
    ${xtensor-python_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${gtest_SOURCE_DIR}/include
    ${NUMPY_LIBRARIES}
)

# Add link libraries
target_link_libraries(hyperpy PRIVATE 
    xtensor
    xtensor-python
    pybind11::module
    Python::Python
    ${NUMPY_LIBRARIES}
)

add_executable(hypergrad_test tests/test_fcnn.cpp)
target_include_directories(hypergrad_test PRIVATE 
  ${CMAKE_SOURCE_DIR}/include
  ${xtensor_SOURCE_DIR}/include 
  ${xtl_SOURCE_DIR}/include
)
target_link_libraries(hypergrad_test PRIVATE 
    gtest_main
    gmock_main
)
add_test(hypergrad_test hypergrad_test)