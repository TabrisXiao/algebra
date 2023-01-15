
#ifndef AA_OPS_TD_
#define AA_OPS_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "AADialect.td"
include "AATypes.td"

class AA_Op<string mnemonic, list<Trait> traits = []>:
    Op<AA_Dialect, mnemonic, traits>;

class AA_ObjDecl <string mnemonic, list<Trait> traits = []>: 
    AA_Op<mnemonic, 
    traits # [
        NoSideEffect,
        DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]>
{
    let summary = "the base operation to define algebra objects";
    //let assemblyFormat = "attr-dict 
}

def AA_AElemDecl : AA_ObjDecl<"AElemDecl", []>{
    let summary = "The operation declaring an abstract algebra element";
    let arguments = (ins FlatSymbolRefAttr:$encoding);
    let results = (outs AA_AElemType:$output);
    let assemblyFormat = [{ attr-dict `:` type(results)}];
}

class AA_BinaryOp <string mnemonic, list<Trait> traits = []>: 
    AA_Op<mnemonic, 
    traits # [
        NoSideEffect,
        DeclareOpInterfaceMethods<InferTypeOpInterface>
    ]>, 
    Arguments<(ins AA_AElemType:$lhs, AA_AElemType:$rhs)>,
    Results<(outs AA_AElemType:$output)>
{
    let summary = "The binary operation for abstract algebra";
}

def AA_Multiply : AA_BinaryOp<"Multiply",[]>
{
    let summary = "The abstract multiplication for two abstract element";
    //let assemblyFormat = "type($lhs) `,` type($rhs) attr-dict";
}

def AA_Add : AA_BinaryOp<"Add",[]>
{
    let summary = "The abstract addition for two abstract element";
}

def AA_Inverse : AA_Op<"Inverse",
    [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Operation return the inverse of the input";
    let arguments = (ins AA_AElemType:$input);
    let results = (outs AA_AElemType:$output);
    let hasCanonicalizer = 1;
    let assemblyFormat = "`(`$input`)` attr-dict `:` type($input) `->`type(results)";
}

def AA_Negative : AA_Op<"Negative", 
    [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Operation return the inverse of input respect to the addition operation";
    let arguments = (ins AA_AElemType:$input);
    let results = (outs AA_AElemType:$output);
    let hasCanonicalizer = 1;
}

#endif //AA_OPS_TD_