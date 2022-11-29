
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "dialect/MLA/type.hpp"
#include "dialect/MLA/dialect.hpp"

void MC::MLA::MLADialect::initialize(){
    registerTypes();
    addOperations<
        #define GET_OP_LIST
        #include "dialect/MLA/generated/op.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "dialect/MLA/generated/op.cpp.inc"