
#ifndef AAPASSES_H_
#define AAPASSES_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace MC::AA{
    std::unique_ptr<mlir::Pass> createAssociationAbstractPass();

    // introduce the pass definition. It has append to the end of all pass creation 
    // functions.
    #define GEN_PASS_CLASSES
    #define GEN_PASS_REGISTRATION
    #include "AA/AAPasses.h.inc"
}

#endif