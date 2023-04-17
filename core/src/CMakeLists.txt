file(GLOB_RECURSE headers
    printer.h
    dgraph.h
    operation.h
    #"${PROJECT_SOURCE_DIR}/core/include/*.h"
)
set(
    sources
    operation.cpp
    dgraph.cpp
    # opBuilder.cpp
    # sdgraph.cpp
    # utility.cpp
)

create_cpp_library(
    NAME logic_graph_lib
    #HDRS
    #${headers}
    SRCS
    ${sources}
    INCLUDES
    ${PROJECT_SOURCE_DIR}/core/include/
    COPTS /EHsc
)