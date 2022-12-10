

#ifndef MLA_DIALECT_TD_
#define MLA_DIALECT_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def MLA_Dialect : Dialect{
    let name = "MLA";
    let cppNamespace = "::MC::MLA";
    let extraClassDeclaration = [{
        private:
            //void registerAttributes();
            //void registerLocationAttributes();
            void registerTypes();
        //public:
            }];
    
    //generate the declarations for parser and printer 
    let useDefaultTypePrinterParser = 1;
}

#endif //MLA_DIALECT_TD_