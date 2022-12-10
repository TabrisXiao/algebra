
#include "dialect/MLA/type.hpp"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "dialect/MLA/generated/type.cpp.inc"

void MC::MLA::MLADialect::registerTypes() {
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "dialect/MLA/generated/type.cpp.inc"
    >();
}