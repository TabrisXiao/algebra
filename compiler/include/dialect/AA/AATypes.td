
#ifndef AA_TYPES_TD_
#define AA_TYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "AADialect.td"

class AA_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<AA_Dialect, name, traits> {
    //the name is the class name in cpp declarations
    // mnemonic is used for parsing and printing
    let mnemonic = typeMnemonic;
}

def AA_AElemType : AA_Type<"AElem", "AElemType">{
    let description = [{
        The type for distinguish different abstract elements.
    }];
    let parameters = (ins
        "::mlir::Attribute":$encoding
    );
    let hasCustomAssemblyFormat = 1;
    //let assemblyFormat = " $encoding ";
}

#endif //AA_TYPES_TD_