cmake_minimum_required(VERSION 3.16)
project(horizon LANGUAGES CXX)
include(CTest)

# options
option(HORIZON_PROFILING "enable profiling features" OFF)
option(HORIZON_BUILD_PYTHON "Build Python bindings" ON)
option(HORIZON_BUILD_EXAMPLES "Build standalone solver examples" ON)
option(HORIZON_TEST_PINOCCHIO "Build optional native Pinocchio integration tests" OFF)
option(HORIZON_TEST_SPOT "Build native Spot contact tests and profiling benchmark" OFF)
option(HORIZON_WITH_PINOCCHIO "Build Pinocchio collision kinematics adapter" OFF)
option(HORIZON_WITH_OPENVDB "Build OpenVDB distance-field support and mesh converter" OFF)
option(HORIZON_WITH_KOKKOS "Build Kokkos collision batches across shooting nodes" OFF)
option(HORIZON_BENCH_KOKKOS_KERNELS "Build optional collision accumulation library comparison" OFF)
option(HORIZON_TEST_KYON_COLLISION "Build integration test against the Kyon URDF" OFF)
set(HORIZON_KYON_URDF "" CACHE FILEPATH "Path to the canonical Kyon URDF used by collision integration tests")

if(${HORIZON_PROFILING})
    add_definitions(-DHORIZON_PROFILING)
endif()

find_package(Eigen3 REQUIRED)
find_package(casadi 3.5.5 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_STANDARD 20)

# ilqr library
add_library(horizon STATIC
    src/functions.cpp
    src/variables.cpp
    src/bounds.cpp)

target_link_libraries(horizon Eigen3::Eigen casadi::casadi)
target_include_directories(horizon PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                         $<INSTALL_INTERFACE:include>)

add_library(ilqr STATIC
    src/ilqr.cpp
    src/ilqr_kkt_solver.cpp
    src/ilqr_forward_pass.cpp
    src/ilqr_backward_pass.cpp
    src/wrapped_function.cpp
    src/profiling.cpp
    src/codegen_function.cpp
    src/iterate_filter.cpp
    src/ilqr_function.cpp
    src/ilqr_casadi_function.cpp
    src/ilqr_state_space.cpp
    src/ilqr_managed.cpp
    src/ilqr_symbolic.cpp
    src/collision.cpp
)

if(HORIZON_WITH_PINOCCHIO)
    find_package(pinocchio REQUIRED)
    target_sources(ilqr PRIVATE src/pinocchio_collision.cpp)
    target_link_libraries(ilqr pinocchio::pinocchio)
    target_compile_definitions(ilqr PUBLIC HORIZON_WITH_PINOCCHIO=1)
endif()

if(HORIZON_WITH_OPENVDB)
    find_package(OpenVDB REQUIRED)
    target_sources(ilqr PRIVATE src/openvdb_collision.cpp)
    target_link_libraries(ilqr OpenVDB::openvdb)
    target_compile_definitions(ilqr PUBLIC HORIZON_WITH_OPENVDB=1)
    add_executable(horizon_mesh_to_sdf src/mesh_to_sdf.cpp)
    target_link_libraries(horizon_mesh_to_sdf PRIVATE OpenVDB::openvdb)
    install(TARGETS horizon_mesh_to_sdf RUNTIME DESTINATION bin)
endif()

if(HORIZON_WITH_KOKKOS)
    find_package(Kokkos 4 REQUIRED)
    target_sources(ilqr PRIVATE src/kokkos_collision.cpp src/device_collision.cpp)
    target_link_libraries(ilqr Kokkos::kokkos)
    target_compile_definitions(ilqr PUBLIC HORIZON_WITH_KOKKOS=1)
    if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
        # Eigen/Pinocchio/CasADi solver code remains host-only. Avoid NVCC's
        # device analysis of those large template translation units.
        get_target_property(ilqr_sources ilqr SOURCES)
        foreach(source IN LISTS ilqr_sources)
            if(NOT source MATCHES "(kokkos_collision|device_collision)\\.cpp$")
                set_source_files_properties(${source} PROPERTIES COMPILE_OPTIONS "--host-only")
            endif()
        endforeach()
    endif()
endif()

# sqp library
add_library(sqp STATIC
    src/wrapped_function.cpp
    src/profiling.cpp
    src/sqp.cpp
)


target_link_libraries(ilqr Eigen3::Eigen casadi::casadi pthread)
target_include_directories(ilqr PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                                      $<INSTALL_INTERFACE:include>)
target_link_libraries(sqp Eigen3::Eigen casadi::casadi ilqr)

install(TARGETS horizon ilqr sqp
    DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)


# python binding
if(HORIZON_BUILD_PYTHON)
set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 REQUIRED)

pybind11_add_module(pyilqr src/pyilqr.cpp)
target_link_libraries(pyilqr PUBLIC ilqr)

pybind11_add_module(pysqp src/pysqp.cpp)
target_link_libraries(pysqp PUBLIC sqp Eigen3::Eigen casadi::casadi)

add_custom_command(TARGET pyilqr POST_BUILD
    COMMAND cp $<TARGET_FILE:pyilqr> ../solvers
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying $<TARGET_FILE:pyilqr> to pkg dir"
)

add_custom_command(TARGET pysqp POST_BUILD
    COMMAND cp $<TARGET_FILE:pysqp> ../solvers
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying $<TARGET_FILE:pysqp> to pkg dir"
)

IF(NOT DEFINED ENV{CONDA_DEFAULT_ENV})
    add_custom_target(generate_python_package
        COMMAND ./copy_external_dependencies.py $<TARGET_FILE:pyilqr> -c -d horizon/external_libs --solib-dst horizon/solvers --solib-rpath ../external_libs
        COMMAND ./copy_external_dependencies.py $<TARGET_FILE:pysqp> -c -d horizon/external_libs --solib-dst horizon/solvers --solib-rpath ../external_libs
        COMMAND ./copy_external_dependencies.py libcasadi_conic_osqp.so -c -f -d horizon/external_libs
        DEPENDS pysqp pyilqr
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../..
        COMMENT "Generating package ${MODULE_NAME}"
        )
ELSE()
    add_custom_target(generate_python_package
        COMMAND ./copy_external_dependencies.py $<TARGET_FILE:pyilqr> -c -d horizon/external_libs --solib-dst horizon/solvers --solib-rpath ../external_libs
        COMMAND ./copy_external_dependencies.py $<TARGET_FILE:pysqp> -c -d horizon/external_libs --solib-dst horizon/solvers --solib-rpath ../external_libs
        DEPENDS pysqp pyilqr
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../..
        COMMENT "Generating package ${MODULE_NAME}"
    )
ENDIF()

add_custom_target(upload_python_package
    COMMAND twine upload dist/*
    DEPENDS generate_python_package
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/../..
    COMMENT "Uploading package ${MODULE_NAME}"
    )

# test executable
endif()
if(HORIZON_BUILD_EXAMPLES)
if(HORIZON_WITH_PINOCCHIO)
    add_executable(kyon_collision_example src/kyon_collision_example.cpp)
    target_link_libraries(kyon_collision_example PRIVATE ilqr pinocchio::pinocchio)
    if(HORIZON_KYON_URDF)
        target_compile_definitions(kyon_collision_example PRIVATE HORIZON_KYON_URDF_PATH=\"${HORIZON_KYON_URDF}\")
    endif()
    install(TARGETS kyon_collision_example RUNTIME DESTINATION bin)
endif()
add_executable(ilqr_mixed_example src/ilqr_mixed_example.cpp)
target_link_libraries(ilqr_mixed_example ilqr)
if(BUILD_TESTING)
    add_test(NAME ilqr_mixed_example COMMAND ilqr_mixed_example)
endif()
add_executable(ilqr_test src/ilqr_test.cpp)
target_link_libraries(ilqr_test ilqr)

add_executable(sqp_test src/sqp_test.cpp)
target_link_libraries(sqp_test sqp)

add_executable(sqp_test2 src/sqp_test2.cpp)
target_link_libraries(sqp_test2 sqp)
endif()


if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
