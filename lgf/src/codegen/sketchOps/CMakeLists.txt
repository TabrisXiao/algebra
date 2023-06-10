
set(sources
    main.cpp 
    #tests/test_aog.h
)
create_cpp_binary(
    NAME sketchCodegen
    SRCS 
    ${sources}

    LINK_LIBS
    lgf_lib
    lgf_codegen_lib
    COPTS /EHsc /W0 /std:c++17
)