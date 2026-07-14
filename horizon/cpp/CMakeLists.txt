cmake_minimum_required(VERSION 3.16)
project(horizon VERSION 1.0)

# pybind11 2.9 still uses CMake's legacy Python discovery modules.
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
endif()


# options
option(HORIZON_PROFILING OFF "enable profiling features")

if(${HORIZON_PROFILING})
    add_definitions(-DHORIZON_PROFILING)
endif()

find_package(Eigen3 REQUIRED)
find_package(casadi REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(casadi_kin_dyn REQUIRED)

#find_package(phase_manager REQUIRED)

#find_package(catkin REQUIRED COMPONENTS phase_manager)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set(CMAKE_CXX_STANDARD 20)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIRS}
    )

# horizon library
add_library(horizon STATIC
    src/variables.cpp
    src/bounds.cpp
    src/functions.cpp
    src/profiling.cpp
    )

target_link_libraries(horizon PUBLIC Eigen3::Eigen casadi::casadi)

add_library(ilqr STATIC
    src/ilqr.cpp
    src/ilqr_kkt_solver.cpp
    src/ilqr_forward_pass.cpp
    src/ilqr_backward_pass.cpp
    src/wrapped_function.cpp
    src/codegen_function.cpp
    src/iterate_filter.cpp
)

# sqp library
add_library(sqp STATIC
    src/wrapped_function.cpp
    src/sqp.cpp
)

target_link_libraries(ilqr PUBLIC pthread horizon)
target_link_libraries(sqp PUBLIC ilqr horizon)

target_include_directories(horizon
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include/${PROJECT_NAME}> #
    )

install(
    TARGETS horizon ilqr sqp
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION lib
)

install(
    DIRECTORY include/horizon include/ilqr include/sqp
    DESTINATION include/${PROJECT_NAME}
    ) #/${PROJECT_NAME}

# python binding
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
add_executable(ilqr_test src/ilqr_test.cpp)
target_link_libraries(ilqr_test ilqr)

add_executable(sqp_test src/sqp_test.cpp)
target_link_libraries(sqp_test sqp)

add_executable(sqp_test2 src/sqp_test2.cpp)
target_link_libraries(sqp_test2 sqp)

#add_executable(classes_test src/classes_test.cpp)
#target_link_libraries(classes_test ilqr horizon yaml-cpp)

#add_executable(phase_manager_test src/phase_manager_test.cpp)
#target_link_libraries(phase_manager_test ilqr horizon yaml-cpp ${catkin_LIBRARIES})

add_subdirectory(tests)

include(cmake/ExportProject.cmake)
export_project()
