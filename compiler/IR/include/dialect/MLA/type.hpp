
#ifndef IR_MLA_TYPE_H_
#define IR_MLA_TYPE_H_
#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/SubElementInterfaces.h"

#include "dialect/MLA/dialect.hpp"

#define GET_TYPEDEF_CLASSES
#include "dialect/MLA/generated/type.hpp.inc"

#endif