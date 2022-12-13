
#ifndef IR_AA_TYPE_H_
#define IR_AA_TYPE_H_
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Types.h"

#include "AA/AADialect.h"

#define GET_TYPEDEF_CLASSES
#include "AA/AATypes.h.inc"

#endif