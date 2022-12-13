set(sources
    main.cpp 
    #test_unit/test_AA.h
    test_frame/test_ops.h
)
create_cpp_binary(
    NAME test_compiler
    SRCS 
    ${sources}

    LINK_LIBS
    compiler_frame_lib
    MLIRAADialectLib
    MLIRIR
    MLIRInferTypeOpInterface
    COPTS /EHsc /W0 /std:c++17
)