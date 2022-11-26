#ifndef MLA_DIALECT_H_
#define MLA_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dialect/MLA/generated/dialect.hpp.inc"

#define GET_TYPEDEF_CLASSES
#include "dialect/MLA/generated/type.hpp.inc"
#define GET_OP_CLASSES
#include "dialect/MLA/generated/op.hpp.inc"
namespace MC::MLA{
    
}
#endif