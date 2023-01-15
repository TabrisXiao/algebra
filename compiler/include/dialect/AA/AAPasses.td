
#ifndef AA_PASSES_TD_
#define AA_PASSES_TD_

include "mlir/Pass/PassBase.td"

def AssociationAbstract : Pass<"aa-association-abstract">{
    let summary = "reduce the algebraic operation numbers based on association law";
    let constructor = "MC::AA::createAssociationAbstractPass()";
    let dependentDialects = [
        "MC::AA::AADialect",
        "mlir::func::FuncDialect"
        ];
}

#endif
