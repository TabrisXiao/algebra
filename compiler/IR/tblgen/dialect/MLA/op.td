
#ifndef MLA_OPS_TD_
#define MLA_OPS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dialect/MLA/dialect.td"
include "dialect/MLA/type.td"

class MLA_Op<string mnemonic, list<Trait> traits = []>:
    Op<MLA_Dialect, mnemonic, traits>;

class MLA_ObjDecl <string mnemonic>: 
    MLA_Op<mnemonic, [NoSideEffect]>{
        let summary = "the base operation to define MLA objects";
        ///let assemblyFormat = "attr-dict 
}

def MLA_TensorBasisDecl : MLA_ObjDecl<"xTensorBasisDecl">{
    let summary = "The operation declaring a tensor";
    let arguments = (ins DenseI32ArrayAttr:$shape,
                    FlatSymbolRefAttr:$encoding);
    let results = (outs MLA_TensorBasisType:$output);
}


//def MLA_TensorDecl : MLA_ObjDecl<"TensorDecl">{
//    let summary = "The operation declaring a tensor";
//    let arguments = (ins xTensorBasisType:$basis);
//    let results = (outs AnyRankedTensor:$output);
//}

//def MLA_ContractionOp : MLA_Op<"ContractionOp">{
//    let summary = "The operation contracting two tensors along a given axis";
//    let arguments = (ins AnyRankedTensor:$lhs,
//                         AnyRankedTensor:$rhs);
//    let results = (outs AnyRankedTensor:$output);
//}

#endif //MLA_OPS_TD_