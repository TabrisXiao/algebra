
#ifndef FRAME_INIT_H_
#define FRAME_INIT_H_
#include "mlir/IR/MLIRContext.h"
#include "AA/AADialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace MC{
void registerDialect(mlir::DialectRegistry & reg);
}

#endif