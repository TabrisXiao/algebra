#include <iostream>
#include "dialect/AA/AADialect.h"
#include "dialect/AA/AATypes.h"
#include "dialect/AA/AAOps.h"

bool test_AA(){
    mlir::MLIRContext context;
    context.getOrLoadDialect<MC::AA::AADialect>();
    mlir::OpBuilder builder(&context);
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    auto symbol = mlir::FlatSymbolRefAttr::get(&context, llvm::StringRef("abel"));
    auto lhs = builder.create<MC::AA::AElemDecl>(
        builder.getUnknownLoc(),
        symbol
    );
    auto rhs = builder.create<MC::AA::AElemDecl>(
        builder.getUnknownLoc(),
        symbol
    );
    auto mul = builder.create<MC::AA::Multiply>(
        builder.getUnknownLoc(),
        lhs,
        rhs
    );
    auto inverse = builder.create<MC::AA::Inverse>(
        builder.getUnknownLoc(),
        lhs);
    inverse->dump();
    return false;
}