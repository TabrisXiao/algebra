
#include "dialect/MLA/type.hpp"
#include "dialect/MLA/dialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "dialect/MLA/generated/type.cpp.inc"

void MC::MLA::MLADialect::registerTypes() {
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "dialect/MLA/generated/type.cpp.inc"
    >();
}