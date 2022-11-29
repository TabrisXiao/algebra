
#include <iostream>
#include "mlir/IR/Builders.h"
#include "dialect/MLA/dialect.hpp"


int main(){
    mlir::MLIRContext context;
    mlir::OpBuilder builder(&context);
    
    return 0;
}