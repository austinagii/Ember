cmake_minimum_required(VERSION 3.14)
project(ember VERSION 1.1.0)

set(EMBER_VERSION_MAJOR 1)
set(EMBER_VERSION_MINOR 1)
set(EMBER_VERSION_PATCH 0)

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

FetchContent_Declare(
  xtensor-blas
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
  GIT_TAG       0.21.0
)

FetchContent_MakeAvailable(gtest xtl xtensor xtensor-blas)

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
  src/ember/ops/matmul.cpp
  src/ember/ops/exp.cpp
  src/ember/ops/utils.cpp
)

# Option for building tests (ON by default for standalone builds)
option(EMBER_BUILD_TESTS "Build ember tests" ${PROJECT_IS_TOP_LEVEL})

if(EMBER_BUILD_TESTS)
    # Enable testing
    enable_testing()
    
    # Test files
    set(EMBER_TESTS
        tests/ember/test_tensor.cpp
        tests/ember/ops/test_sub.cpp
        tests/ember/ops/test_add.cpp
        tests/ember/ops/test_mul.cpp
        tests/ember/ops/test_div.cpp
        tests/ember/ops/test_matmul.cpp
        tests/ember/ops/test_exp.cpp
        tests/ember/test_readme.cpp
    )

    # Create test executable
    add_executable(ember_test ${EMBER_TESTS})
    target_link_libraries(ember_test PRIVATE 
        ember
        gtest_main 
        gmock_main
    )

    add_test(NAME ember_test COMMAND ember_test)
endif()

# Configure version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ember/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/ember/version.h
)

# Create static library
add_library(ember STATIC ${EMBER_SOURCES})
target_include_directories(ember 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(ember PUBLIC xtl xtensor xtensor-blas)

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

# Update install command to include the generated version header
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/ember/version.h 
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ember)
