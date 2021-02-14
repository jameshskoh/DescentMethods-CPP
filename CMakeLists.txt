cmake_minimum_required(VERSION 3.17)
project(DescentMethods)

set(CMAKE_CXX_STANDARD 14)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(DescentMethods main.cpp descentmethods.cpp descentmethods.h)
target_link_libraries(DescentMethods ${CONAN_LIBS})
