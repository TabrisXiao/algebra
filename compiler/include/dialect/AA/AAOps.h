#ifndef AA_OP_H_
#define AA_OP_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"


#include "dialect/AA/AADialect.h"
#include "dialect/AA/AATypes.h"

#define GET_OP_CLASSES
#include "dialect/AA/AAOps.h.inc"
#endif