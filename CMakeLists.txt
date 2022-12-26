
cmake_minimum_required(VERSION 3.13.4)

project(algebra LANGUAGES CXX C)
#set(CMAKE_BINARY_DIR output)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

set(MLIR_BUILD_DIR "D:/projects/llvm-project/build")
set(MLIR_DIR ${MLIR_BUILD_DIR}/lib/cmake/mlir)
set(LLVM_DIR ${MLIR_BUILD_DIR}/lib/cmake/llvm)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(MLIR_TABLEGEN_EXE ${MLIR_BUILD_DIR}/Debug/bin/)
include(MLIRloader)
include(MLIRUtils)
include(builder)

set(algebra_dialect_library "algebra_dialect_library")
set(compiler_root ${CMAKE_SOURCE_DIR}/compiler)
add_subdirectory(${compiler_root})
add_subdirectory(test)
add_subdirectory(core)