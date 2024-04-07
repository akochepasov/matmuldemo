cmake_minimum_required (VERSION 3.20)
message(STATUS "CMAKE_VERSION: ${CMAKE_VERSION}")
message(STATUS "CMAKE_COMMAND: ${CMAKE_COMMAND}") 
include(FetchContent)


# CMAKE settings
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CPP settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (UNIX)
	# Using GCC
	# add_compile_options(-msse4.2)
	add_compile_options(-mavx2)
	# add_compile_options(-DMKL_ILP64 -m64)
	#add_compile_options(-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vbmi -mavx512vbmi2 -mavx512vl)
	add_compile_options(-Wall -Wno-unknown-pragmas)
	#add_compile_options(-frecord-gcc-switches)
else()
    # Using MSVC
	# add_compile_options crashes nvcc
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") # for googlebenchmark
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /openmp:llvm")
endif()

# CUDA settings
set(CUDA_VERBOSE_BUILD OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 60) # 70 80
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler")   # Allow VS preview
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -src-in-ptx --cudart shared")


project(matmuldemo LANGUAGES CXX CUDA)


# Enable eigen
set(BUILD_TESTING OFF)	
# set(EIGEN_BUILD_TESTING OFF)
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
find_package(OpenMP REQUIRED)

find_package(CUDAToolkit REQUIRED)


# Add source to this project's executable.
add_executable (matmuldemo "matmuldemo.cpp" "matmuldemo.h" "matmuldemo.cu")

target_link_libraries(matmuldemo PRIVATE Eigen3::Eigen)
target_link_libraries(matmuldemo PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(matmuldemo PRIVATE benchmark::benchmark)
target_link_libraries(matmuldemo PRIVATE CUDA::cublas CUDA::cudart)
