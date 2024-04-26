
cmake_minimum_required(VERSION 3.13.4)

project(algebra LANGUAGES CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_BINARY_DIR output)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(MSVC)
    set(COMPILER_OPTIONS /EHsc /W0 /std:c++17)
elseif(APPLE)
    set(COMPILER_OPTIONS -std=c++17 -Wno-unused-variable)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(builder)

add_subdirectory(lgf)
add_subdirectory(test)
