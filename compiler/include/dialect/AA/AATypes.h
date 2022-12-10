
#ifndef IR_AA_TYPE_H_
#define IR_AA_TYPE_H_
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/SubElementInterfaces.h"

#include "dialect/AA/AADialect.h"

#define GET_TYPEDEF_CLASSES
#include "dialect/AA/AATypes.h.inc"

#endif