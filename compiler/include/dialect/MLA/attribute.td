
#ifndef MLA_ATTRIBUTE_TD_
#define MLA_ATTRIBUTE_TD_

include "mlir/IR/AttrTypeBase.td"
include "dialect/MLA/op.td"

class MLA_Attr <string name, list<Trait> traits = [], string baseCppClass = "::mlir::Attribute"> : AttrDef<MLA_Dialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

//def MLA_SymbolicIndexAttr : MLA_Attr<"xSymbolicIndex">{
//    let summary = "An attribute containing a symbol as index";
//    let parameters = (ins FlatSymbolRefAttr:$name);
//    //let storageType = "MLA::SymbolicIndexAttr";
//}

#endif
