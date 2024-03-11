cmake_minimum_required (VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")


# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

set(CMAKE_CXX_STANDARD 17)

# CUDA settings
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")   # Allow VS preview
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(matmuldemo LANGUAGES C CXX CUDA)


include(FetchContent)

# Enable eigen
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
set(EIGEN_BUILD_PKGCONFIG OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(eigen)


# Enable google benchmark
set(BENCHMARK_ENABLE_GTEST_TESTS OFF)
set(BENCHMARK_ENABLE_TESTING OFF)
set(BENCHMARK_ENABLE_LTO OFF)
set(HAVE_STD_REGEX ON)

FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.7.1
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(googlebenchmark)


# Enable OpenMP
find_package(OpenMP)


# Add source to this project's executable.
add_executable (matmuldemo "matmuldemo.cpp" "matmuldemo.h" "matmuldemo.cu")

target_link_libraries(matmuldemo PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(matmuldemo PRIVATE Eigen3::Eigen)
target_link_libraries(matmuldemo PRIVATE benchmark::benchmark)
target_link_libraries(matmuldemo PRIVATE cublas cudart)
