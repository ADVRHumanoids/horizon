cmake_minimum_required(VERSION 3.22)
project(horizon_nvblox_probe LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDAToolkit 13 REQUIRED)
find_package(Eigen3 REQUIRED)
set(NVBLOX_SOURCE "" CACHE PATH "nvblox source checkout")
set(NVBLOX_BUILD "" CACHE PATH "Existing nvblox build with nvblox_lib built")
add_executable(probe_nvblox probe_nvblox.cu)
target_include_directories(probe_nvblox SYSTEM PRIVATE
    ${NVBLOX_SOURCE}/nvblox/include
    ${NVBLOX_BUILD}/_deps/ext_stdgpu-src/src
    ${NVBLOX_BUILD}/_deps/ext_stdgpu-build/src/stdgpu/include
    ${NVBLOX_BUILD}/_deps/ext_glog-build
    ${NVBLOX_BUILD}/_deps/ext_glog-src/src
    ${NVBLOX_BUILD}/_deps/ext_gflags-build/include)
target_compile_definitions(probe_nvblox PRIVATE CUB_WRAPPED_NAMESPACE=nvblox
    THRUST_DISABLE_ABI_NAMESPACE THRUST_IGNORE_ABI_NAMESPACE_ERROR)
target_compile_options(probe_nvblox PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda>)
target_link_libraries(probe_nvblox PRIVATE Eigen3::Eigen CUDA::cudart
    ${NVBLOX_BUILD}/nvblox/libnvblox_lib.so
    ${NVBLOX_BUILD}/nvblox/libnvblox_gpu_hash.a
    ${NVBLOX_BUILD}/_deps/ext_stdgpu-build/bin/libstdgpu.a
    ${NVBLOX_BUILD}/_deps/ext_glog-build/libglog.a
    ${NVBLOX_BUILD}/_deps/ext_gflags-build/libgflags_nothreads.a unwind)
set(CUROBO_SOURCE "" CACHE PATH "Optional cuRoboV2 source for GPU scene exchange")
set(CUROBO_BUILD "" CACHE PATH "Optional native cuRoboV2 build")
if(CUROBO_SOURCE AND CUROBO_BUILD)
    target_compile_definitions(probe_nvblox PRIVATE HORIZON_CUROBO_SCENE=1)
    target_include_directories(probe_nvblox PRIVATE ${CUROBO_SOURCE}/cpp_runtime/robot/include)
    target_link_libraries(probe_nvblox PRIVATE ${CUROBO_BUILD}/libcurobo_robot_runtime.so)
endif()
set_target_properties(probe_nvblox PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
