
#ifndef MLA_OPS
#define MLA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MLA_Dialect : Dialect{
    let name = "MLA";
    let cppNamespace = "MC::MLA";
}

class MLA_Op<string mnemonic, list<Trait> traits = []>:
    Op<MLA_Dialect, mnemonic, traits>;



#endif //MLA_OPS