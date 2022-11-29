
#ifndef MLA_OPS
#define MLA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "dialect/MLA/type.td"


def MLA_Dialect : Dialect{
    let name = "MLA";
    let cppNamespace = "::MC::MLA";
    let extraClassDeclaration = [{
        private:
            //void registerAttributes();
            //void registerLocationAttributes();
            void registerTypes();
        public:
            }];
}

class MLA_Op<string mnemonic, list<Trait> traits = []>:
    Op<MLA_Dialect, mnemonic, traits>;

class MLA_ObjDeclOp <string mnemonic>: 
    MLA_Op<mnemonic, [NoSideEffect]>{
        let summary = "the base operation to define MLA objects";
        ///let assemblyFormat = "attr-dict 
}

def MLA_TensorBasisDecl : MLA_ObjDeclOp<"TensorDeclOp">{
    let summary = "The operation declaring a tensor";
    let arguments = (ins ArrayAttr :$shape);
    let results = (outs MLA_TensorBasisType:$output);
}


//def MLA_TensorDeclOp : MLA_ObjDeclOp<"TensorDeclOp">{
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

#endif //MLA_OPS