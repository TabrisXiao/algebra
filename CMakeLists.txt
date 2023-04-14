
cmake_minimum_required(VERSION 3.13.4)

project(algebra LANGUAGES CXX C)
#set(CMAKE_BINARY_DIR output)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(builder)

add_subdirectory(test)
add_subdirectory(core)