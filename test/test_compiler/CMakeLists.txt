
create_cpp_binary(
    NAME test_compiler
    SRCS main.cpp test_unit/test_AA.h
    LINK_LIBS
    MLIRAADialectLib
    MLIRIR
    MLIRInferTypeOpInterface
    COPTS /EHsc /W0 /std:c++17
)