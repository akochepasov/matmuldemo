﻿cmake_minimum_required (VERSION 3.20)

# CMAKE settings
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
	# Release is the default configuration
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "CMAKE_VERSION:    ${CMAKE_VERSION}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_GENERATOR:  ${CMAKE_GENERATOR}")
message(STATUS "CMAKE_COMMAND:    ${CMAKE_COMMAND}")

include(FetchContent)
include(ExternalProject)

# CPP settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (UNIX)
	# add_compile_options(-msse4.2)
	add_compile_options(-mavx2)
	# add_compile_options(-DMKL_ILP64 -m64)
	#add_compile_options(-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
	add_compile_options(-Wall -Wno-unknown-pragmas)
	#add_compile_options(-frecord-gcc-switches)
else()
	# add_compile_options crashes nvcc, use flags
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")                  # for googlebenchmark
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")   # Use OpenMP 4
endif(UNIX)

# CUDA settings
set(CUDA_VERBOSE_BUILD ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 61) # 70 80   # https://developer.nvidia.com/cuda-gpus
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")   # Allow VS preview

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("Debug mode")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -src-in-ptx --cudart shared")
endif(CMAKE_BUILD_TYPE STREQUAL "Debug")


project(matmuldemo LANGUAGES CXX CUDA)


# Enable eigen
set(BUILD_TESTING OFF)
# set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_MPL2_ONLY ON)
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
  GIT_TAG v1.8.5
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(googlebenchmark)

# Enable OpenMP
find_package(OpenMP REQUIRED)

# Enable CUDA
find_package(CUDAToolkit REQUIRED)

# Enable CUTLASS
set(CUTLASS_ENABLE_CUBLAS OFF)
set(CUTLASS_ENABLE_CUDNN OFF)
set(CUTLASS_ENABLE_HEADERS_ONLY ON)
set(CUTLASS_UNITY_BUILD_ENABLED OFF)
set(CUTLASS_NVCC_ARCHS 61)
set(CUTLASS_LIBRARY_OPERATIONS=gemm)

FetchContent_Declare(
  cutlass
  GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
  GIT_TAG v3.5.1
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(cutlass)

# Enable pytorch (2.3)
if (UNIX)
  set(LIBTORCH_URL    https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu121.zip)
else()
  set(LIBTORCH_URL    https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.4.1%2Bcu121.zip)
endif(UNIX)

if (DEFINED ENV{DEPENDENCES_ROOT})
  set(LIBTORCH_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/libtorch-src)
else()
  set(LIBTORCH_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/libtorch-src)
endif()

# torch-cpp requres nvToolsExt, which is from CUDA legacy toolkit 11.8
# The toolkit should be installed separately
# If nvToolsExt not found on Windows after installation, use the search below
#if(WIN32)
#  # nvtools can be installed outside the CUDA toolkit directory
#   # so prefer the NVTOOLSEXT_PATH windows only environment variable
#  # set(ENV{NVTOOLSEXT_PATH} "path/to/cuda/NvToolsExt/")
#   # In addition on windows the most common name is nvToolsExt64_1
#   find_library(CUDA_nvToolsExt_LIBRARY
#     NAMES nvToolsExt64_1 nvToolsExt64 nvToolsExt
#     PATHS ENV NVTOOLSEXT_PATH
#           ENV CUDA_PATH
#     PATH_SUFFIXES lib/x64 lib
#   )
#endif()

FetchContent_Declare(libtorch
  URL ${LIBTORCH_URL}
  SOURCE_DIR ${LIBTORCH_SOURCE_DIR}
)
FetchContent_MakeAvailable(libtorch)

find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR} NO_DEFAULT_PATH)
add_compile_options(${TORCH_CXX_FLAGS})
# add_compile_options("-D_GLIBCXX_USE_CXX11_ABI=1 -Llibtorch/lib -ltorch -ltorch_cpu -lc10")
# add_compile_options("-Llibtorch/lib -ltorch -ltorch_cpu -lc10 -liomp5m")
# target_link_directories(matmuldemo libtorch/lib)
# target_link_libraries(matmuldemo PRIVATE iomp5m)
#add_link_options("-Llibtorch/lib -ltorch -ltorch_cpu -lc10")

# Add source to this project's executable.
add_executable (matmuldemo "matmuldemo.cpp" "matmuldemo.h" "matmuldemo.cu" "matmuldemo_torch.cu")

target_link_libraries(matmuldemo PRIVATE Eigen3::Eigen)
target_link_libraries(matmuldemo PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(matmuldemo PRIVATE benchmark::benchmark)
target_link_libraries(matmuldemo PRIVATE CUDA::cublas CUDA::cudart)
target_link_libraries(matmuldemo PRIVATE CUTLASS)
target_link_libraries(matmuldemo PRIVATE torch)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET matmuldemo
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:matmuldemo>)
endif(MSVC)
