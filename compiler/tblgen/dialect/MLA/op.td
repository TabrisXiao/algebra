
#ifndef MLA_OPS
#define MLA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MLA_Dialect : Dialect{
    let name = "mla";
    let cppNamespace = "SC::MLA";
}

#endif //MLA_OPS