
set(sources
    main.cpp 
    #tests/test_aog.h
)
create_cpp_binary(
    NAME test_poc
    SRCS 
    ${sources}

    LINK_LIBS
    lgf_graph_core
    lgf_qft
    INCLUDES
    COPTS ${COMPILER_OPTIONS}
)