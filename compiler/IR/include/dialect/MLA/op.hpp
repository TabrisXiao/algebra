#ifndef MLA_OP_H_
#define MLA_OP_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dialect/MLA/dialect.hpp"
#include "dialect/MLA/type.hpp"

#define GET_OP_CLASSES
#include "dialect/MLA/generated/op.hpp.inc"
#endif