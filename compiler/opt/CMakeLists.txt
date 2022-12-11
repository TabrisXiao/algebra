get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRIR
        MLIROptLib
        compiler_frame_lib
)

add_llvm_executable(mc-opt mc-opt.cpp)
target_compile_options(mc-opt PRIVATE /W4)
llvm_update_compile_flags(mc-opt)
target_link_libraries(mc-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(mc-opt)