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
  src/ember/ops/utils.cpp
)

# Test files
set(EMBER_TESTS
  tests/ember/test_tensor.cpp
  tests/ember/ops/test_sub.cpp
  tests/ember/ops/test_add.cpp
  tests/ember/ops/test_mul.cpp
  tests/ember/ops/test_div.cpp
  
  tests/ember/test_readme.cpp
)

# Create static library
add_library(ember STATIC ${EMBER_SOURCES})
target_include_directories(ember 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(ember PUBLIC xtl xtensor)

# Create test executable
add_executable(ember_test ${EMBER_TESTS})
target_link_libraries(ember_test PRIVATE 
    ember
    gtest_main 
    gmock_main
)

add_test(NAME ember_test COMMAND ember_test)

# Installation configuration
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ember)

# Install library and headers
install(TARGETS ember
    EXPORT ember-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT ember-targets
    FILE ember-targets.cmake
    NAMESPACE ember::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create and install config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ember-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ember-config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ember-config.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)
