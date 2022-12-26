
#include <iostream>
#include "init.h"
#include "AA/AADialect.h"
#include "AA/AATypes.h"
#include "AA/AAPasses.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#define EXIT_FAILURE 1
#define EXIT_SUCCESS 0

int main(int argc, char* argv[])
{
    //DiagnosticEngine& engine = ctx->getDiagEngine();
    mlir::registerAllPasses();
    mlir::DialectRegistry reg;
    MC::registerDialect(reg);
    MC::AA::registerAAPasses();

    return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "MC OPT Test tool", reg, false));
}