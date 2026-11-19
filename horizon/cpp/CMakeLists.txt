cmake_minimum_required(VERSION 3.16)
project(horizon LANGUAGES CXX)
include(CTest)

# options
option(HORIZON_PROFILING "enable profiling features" OFF)
option(HORIZON_BUILD_PYTHON "Build Python bindings" ON)
option(HORIZON_BUILD_EXAMPLES "Build standalone solver examples" ON)
option(HORIZON_TEST_PINOCCHIO "Build optional native Pinocchio integration tests" OFF)
option(HORIZON_TEST_SPOT "Build native Spot contact tests and profiling benchmark" OFF)
option(HORIZON_WITH_PINOCCHIO "Build native Pinocchio kinematics and collision adapters" OFF)
option(HORIZON_WITH_OPENVDB "Build OpenVDB distance-field support and mesh converter" OFF)
option(HORIZON_WITH_KOKKOS "Build Kokkos collision batches across shooting nodes" OFF)
option(HORIZON_BENCH_KOKKOS_KERNELS "Build optional collision accumulation library comparison" OFF)
option(HORIZON_TEST_KYON_COLLISION "Build integration test against the Kyon URDF" OFF)

if(${HORIZON_PROFILING})
    add_definitions(-DHORIZON_PROFILING)
endif()

find_package(Eigen3 REQUIRED)
find_package(casadi 3.5.5 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tinyxml2 REQUIRED)

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
    src/ilqr_globalization.cpp
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
    src/collision_clearance.cpp
    src/collision_model_io.cpp
)

if(HORIZON_WITH_PINOCCHIO)
    find_package(pinocchio REQUIRED)
    target_sources(ilqr PRIVATE src/pinocchio_kinematics.cpp src/pinocchio_robot.cpp src/pinocchio_robot_dynamics.cpp src/pinocchio_contact.cpp src/pinocchio_collision.cpp src/pinocchio_sphere_clearance.cpp)
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
    target_sources(ilqr PRIVATE src/kokkos_collision.cpp src/device_collision.cpp src/kokkos_runtime.cpp)
    target_link_libraries(ilqr Kokkos::kokkos)
    target_compile_definitions(ilqr PUBLIC HORIZON_WITH_KOKKOS=1)
    if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
        # Eigen/Pinocchio/CasADi solver code remains host-only. Avoid NVCC's
        # device analysis of those large template translation units.
        if(HORIZON_WITH_OPENVDB)
            target_compile_options(horizon_mesh_to_sdf PRIVATE --host-only)
        endif()
        get_target_property(ilqr_sources ilqr SOURCES)
        foreach(source IN LISTS ilqr_sources)
            if(NOT source MATCHES "(kokkos_collision|device_collision|kokkos_runtime)\\.cpp$")
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


target_link_libraries(ilqr Eigen3::Eigen casadi::casadi pthread tinyxml2::tinyxml2 yaml-cpp)
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

pybind11_add_module(pyilqr src/pyilqr.cpp src/pyilqr_functions.cpp src/pyilqr_collision.cpp)
if(HORIZON_WITH_PINOCCHIO)
    target_sources(pyilqr PRIVATE src/pyilqr_pinocchio.cpp)
endif()
target_link_libraries(pyilqr PUBLIC ilqr)
if(HORIZON_WITH_KOKKOS AND Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    target_compile_options(pyilqr PRIVATE --host-only)
endif()

if(BUILD_TESTING)
    if(PYTHON_EXECUTABLE)
        set(horizon_test_python "${PYTHON_EXECUTABLE}")
    elseif(Python_EXECUTABLE)
        set(horizon_test_python "${Python_EXECUTABLE}")
    else()
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(horizon_test_python "${Python3_EXECUTABLE}")
    endif()
    add_test(NAME pythonIlqrInterop COMMAND ${CMAKE_COMMAND} -E env
        "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
        "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1" "MPLBACKEND=Agg"
        "${horizon_test_python}" -m unittest horizon.tests.test_ilqr_python_interop horizon.tests.test_native_model horizon.tests.test_native_frame_kinematics horizon.tests.test_native_robot horizon.tests.test_native_problem horizon.tests.test_collision_helpers -v)
    set_tests_properties(pythonIlqrInterop PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." LABELS "python;ilqr" TIMEOUT 120)
endif()

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
configure_file(examples/example_paths.h.in ${CMAKE_CURRENT_BINARY_DIR}/example_paths.h @ONLY)
if(BUILD_TESTING AND HORIZON_BUILD_PYTHON)
    execute_process(COMMAND "${horizon_test_python}" -c "import viser, yourdfpy, trimesh, matplotlib"
                    RESULT_VARIABLE horizon_visual_dependencies OUTPUT_QUIET ERROR_QUIET)
    function(horizon_add_visual_test lesson)
        if(horizon_visual_dependencies EQUAL 0)
            add_test(NAME python_native_${lesson}_visualize COMMAND ${CMAKE_COMMAND} -E env
                "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
                "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1" "MPLBACKEND=Agg"
                "${horizon_test_python}" -m horizon.tests.run_native_example
                horizon.examples.native.${lesson} --visualize --visualization-seconds 0.2
                --viewer-port 0 --quiet)
            set_tests_properties(python_native_${lesson}_visualize PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.."
                LABELS "python;examples;visualization" TIMEOUT 300 RUN_SERIAL TRUE)
        endif()
    endfunction()
endif()
foreach(lesson IN ITEMS 01_symbolic 02_parameters)
add_executable(native_${lesson} examples/${lesson}.cpp)
target_link_libraries(native_${lesson} PRIVATE ilqr)
target_include_directories(native_${lesson} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
if(HORIZON_WITH_KOKKOS AND Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
    target_compile_options(native_${lesson} PRIVATE --host-only)
endif()
if(BUILD_TESTING)
    add_test(NAME native_${lesson} COMMAND native_${lesson})
    if(HORIZON_BUILD_PYTHON)
        add_test(NAME python_native_${lesson} COMMAND ${CMAKE_COMMAND} -E env
            "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
            "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1"
            "${horizon_test_python}" -m horizon.tests.run_native_example horizon.examples.native.${lesson})
        set_tests_properties(python_native_${lesson} PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." TIMEOUT 120)
        horizon_add_visual_test(${lesson})
    endif()
endif()
endforeach()
if(BUILD_TESTING)
    add_test(NAME native_cli_options COMMAND native_01_symbolic
        --nodes 20 --threads 1 --max-iterations 30 --dt 0.1)
    if(HORIZON_BUILD_PYTHON)
        add_test(NAME python_native_cli_options COMMAND ${CMAKE_COMMAND} -E env
            "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
            "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1"
            "${horizon_test_python}" -m horizon.tests.run_native_example horizon.examples.native.01_symbolic
            --nodes 20 --threads 1 --max-iterations 30 --dt 0.1)
        set_tests_properties(python_native_cli_options PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." TIMEOUT 120)
    endif()
endif()
if(HORIZON_WITH_PINOCCHIO)
    foreach(lesson IN ITEMS 03_robot_kinematics 04_self_collision 05_environment_primitive 07_whole_body_ik 08_whole_body_self_collision 09_contact_forward 10_contact_inverse)
    add_executable(native_${lesson} examples/${lesson}.cpp)
    target_link_libraries(native_${lesson} PRIVATE ilqr pinocchio::pinocchio)
    target_include_directories(native_${lesson} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    if(HORIZON_WITH_KOKKOS AND Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
        target_compile_options(native_${lesson} PRIVATE --host-only)
    endif()
    if(BUILD_TESTING)
        add_test(NAME native_${lesson} COMMAND native_${lesson})
        if(HORIZON_BUILD_PYTHON)
            add_test(NAME python_native_${lesson} COMMAND ${CMAKE_COMMAND} -E env
                "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
                "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1"
                "${horizon_test_python}" -m horizon.tests.run_native_example horizon.examples.native.${lesson})
            set_tests_properties(python_native_${lesson} PROPERTIES
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." TIMEOUT 120)
            horizon_add_visual_test(${lesson})
        endif()
    endif()
    endforeach()
    if(BUILD_TESTING)
        set_tests_properties(native_08_whole_body_self_collision PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
        if(HORIZON_BUILD_PYTHON)
            set_tests_properties(python_native_08_whole_body_self_collision PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
        endif()
    endif()
    if(HORIZON_WITH_KOKKOS)
        add_executable(native_11_accelerated_collision examples/11_accelerated_collision.cpp)
        target_link_libraries(native_11_accelerated_collision PRIVATE ilqr pinocchio::pinocchio Kokkos::kokkos)
        target_include_directories(native_11_accelerated_collision PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        if(Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
            target_compile_options(native_11_accelerated_collision PRIVATE --host-only)
        endif()
        if(BUILD_TESTING)
            add_test(NAME native_11_accelerated_collision COMMAND native_11_accelerated_collision)
            set_tests_properties(native_11_accelerated_collision PROPERTIES TIMEOUT 300 RUN_SERIAL TRUE
                ENVIRONMENT "OMP_NUM_THREADS=1;OMP_PROC_BIND=false")
            if(HORIZON_BUILD_PYTHON)
                add_test(NAME python_native_11_accelerated_collision COMMAND ${CMAKE_COMMAND} -E env
                    "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
                    "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1"
                    "${horizon_test_python}" -m horizon.tests.run_native_example horizon.examples.native.11_accelerated_collision)
                set_tests_properties(python_native_11_accelerated_collision PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." TIMEOUT 300 RUN_SERIAL TRUE)
                horizon_add_visual_test(11_accelerated_collision)
            endif()
        endif()
    endif()
    if(HORIZON_WITH_OPENVDB)
        add_executable(native_06_environment_mesh examples/06_environment_mesh.cpp)
        target_link_libraries(native_06_environment_mesh PRIVATE ilqr pinocchio::pinocchio)
        target_include_directories(native_06_environment_mesh PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        if(HORIZON_WITH_KOKKOS AND Kokkos_ENABLE_CUDA AND CMAKE_CXX_COMPILER MATCHES "nvcc_wrapper")
            target_compile_options(native_06_environment_mesh PRIVATE --host-only)
        endif()
        if(BUILD_TESTING)
            set(example_mesh_sdf "${CMAKE_CURRENT_BINARY_DIR}/example_obstacle.vdb")
            add_test(NAME native_mesh_sdf_setup COMMAND horizon_mesh_to_sdf
                "${CMAKE_CURRENT_SOURCE_DIR}/examples/data/obstacle.stl" "${example_mesh_sdf}" 0.025 16)
            set_tests_properties(native_mesh_sdf_setup PROPERTIES FIXTURES_SETUP native_mesh_sdf)
            add_test(NAME native_06_environment_mesh COMMAND native_06_environment_mesh)
            add_test(NAME native_06_mesh_override COMMAND native_06_environment_mesh --sdf "${example_mesh_sdf}")
            set_tests_properties(native_06_mesh_override PROPERTIES FIXTURES_REQUIRED native_mesh_sdf)
            if(HORIZON_BUILD_PYTHON)
                add_test(NAME python_native_06_environment_mesh COMMAND ${CMAKE_COMMAND} -E env
                    "HORIZON_PYILQR_EXTENSION=$<TARGET_FILE:pyilqr>"
                    "OMP_NUM_THREADS=1" "OPENBLAS_NUM_THREADS=1"
                    "${horizon_test_python}" -m horizon.tests.run_native_example horizon.examples.native.06_environment_mesh)
                set_tests_properties(python_native_06_environment_mesh PROPERTIES
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../.." TIMEOUT 120)
                horizon_add_visual_test(06_environment_mesh)
            endif()
        endif()
    endif()
    add_executable(kyon_collision_example src/kyon_collision_example.cpp)
    target_link_libraries(kyon_collision_example PRIVATE ilqr pinocchio::pinocchio)
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
