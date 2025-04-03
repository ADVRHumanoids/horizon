cmake_minimum_required(VERSION 3.10)
include(ExternalProject)

set(PROJECTNAME tests)
project(${PROJECTNAME})

find_package(Eigen3 REQUIRED)
find_package(casadi 3.5.5 REQUIRED)

# compite GTest
find_package(GTest QUIET)
if (NOT GTEST_FOUND)
    set(gTestSource "/usr/src/gtest")
    ExternalProject_Add(GTest-ext SOURCE_DIR ${gTestSource}
                                  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external"
                                  INSTALL_COMMAND "")
    set(GTEST_LIB_DIRS "${CMAKE_CURRENT_BINARY_DIR}/external/src/GTest-ext-build/")
    set(GTEST_INCLUDE_DIRS ${gTestSource})
    set(GTEST_BOTH_LIBRARIES gtest gtest_main)
endif()

include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIB_DIRS})

# add local include
include_directories(${CMAKE_CURRENT_SOURCE_DIR} )
enable_testing()

SET(TestLibs ilqr ${GTEST_BOTH_LIBRARIES} pthread Eigen3::Eigen casadi)


set(HORIZON_CPP_TESTS
                        testCasadiUtils

   )

ADD_EXECUTABLE(testCasadiUtils testCasadiUtils.cpp)
TARGET_LINK_LIBRARIES(testCasadiUtils ${TestLibs})
add_dependencies(testCasadiUtils ilqr)
add_test(NAME testCasadiUtils COMMAND testCasadiUtils)

ADD_EXECUTABLE(testQr testQr.cpp)
TARGET_LINK_LIBRARIES(testQr ${TestLibs})
add_test(NAME testQr COMMAND testQr)

ADD_EXECUTABLE(testIlqr testIlqr.cpp)
TARGET_LINK_LIBRARIES(testIlqr ${TestLibs})
add_test(NAME testIlqr COMMAND testIlqr)
