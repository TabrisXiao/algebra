
#include <iostream>
#include "mlir/IR/Builders.h"
#include "dialect/MLA/op.hpp"


int main(){
    mlir::MLIRContext context;
    mlir::OpBuilder builder(&context);
    return 0;
}