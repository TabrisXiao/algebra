
#ifndef MLA_TYPES_TD_
#define MLA_TYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/SubElementInterfaces.td"
include "dialect/MLA/dialect.td"

class MLA_Type<string name, string typeMnemonic, list<Trait> traits = []>
        : TypeDef<MLA_Dialect, name, traits> {
    //the name is the class name in cpp declarations
    // mnemonic is used for parsing and printing
    let mnemonic = typeMnemonic;
}

def MLA_TensorBasisType : MLA_Type<"xTensorBasis", "tensorBasis">{
    let description = [{
        The TensorBasisType is used to identify are two bases are the same or not.
        A tensor, as direct product of several linear operators for
    }];
    let parameters = (ins
        ArrayRefParameter<"int32_t">:$shape,
        "::mlir::Attribute":$encoding
    );
    let assemblyFormat = "`<` $shape `,` $encoding `>`";
}

def MLA_TensorType : MLA_Type<"xTensor", "tensor">{
    let parameters = (ins
        "MC::MLA::xTensorBasisType":$basis
    );
    let assemblyFormat = "`<` $basis `>`";
}

#endif //MLA_TYPES_TD_