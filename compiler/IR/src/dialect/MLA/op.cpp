#include "dialect/MLA/op.hpp"

#define GET_OP_CLASSES
#include "dialect/MLA/generated/op.cpp.inc"


void MC::MLA::MLADialect::initialize(){
    addOperations<
    #define GET_OP_LIST
    #include "dialect/MLA/generated/op.cpp.inc"
    >();
}