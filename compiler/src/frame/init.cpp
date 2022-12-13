
#include "init.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

void MC::registerDialect(mlir::DialectRegistry & reg)
{
    reg.insert<MC::AA::AADialect, mlir::func::FuncDialect>();
}