
#include <iostream>
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "dialect/MLA/dialect.hpp"


int main(){
    mlir::MLIRContext context;
    mlir::OpBuilder builder(&context);
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    std::vector<int64_t> buf{2,4,2};
    llvm::SmallVector<int, 3> shape;
    shape.push_back(2);
    shape.push_back(2);
    shape.push_back(2);
    auto symbol = mlir::FlatSymbolRefAttr::get(&context, llvm::StringRef("test"));
    auto type = mlir::xTensorBasisType::get(&context, llvm::makeArrayRef(buf), symbol);
    auto shp = mlir::DenseI64ArrayAttr::get(&context, llvm::makeArrayRef(buf));
    auto tensor = builder.create<MC::MLA::TensorBasisDecl>(
        builder.getUnknownLoc(),
        type,
        shp
    );
    tensor->dump();
    //mlir::ArrayAttr::get(&context, shape);
    return 0;
}