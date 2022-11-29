
#ifndef MLA_TYPES_TD
#define MLA_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"

class MLA_Type<string name, list<Trait> traits = []>
        : TypeDef<Builtin_Dialect, name, traits> {
    let mnemonic = ?;// name;
}

def MLA_TensorBasisType : MLA_Type<"xTensorBasis">{
    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Attribute":$encoding
    );
    //let assemblyFormat = "`<` $shape `>`";
}

//def MLA_TensorType : MLA_Type<"xTensor">{
//    let parameters = (ins
//        ArrayRefParameter<"int64_t">:$shape,
//        "Basis":$basis,
//        "Attribute":$encoding
//    );
//    //let assemblyFormat = "`<` $basis `>`";111
//}

#endif
