

#ifndef AA_DIALECT_TD_
#define AA_DIALECT_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

// abstract algebra dialect
def AA_Dialect : Dialect{
    let name = "AA";
    let cppNamespace = "::MC::AA";
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

#endif //AA_DIALECT_TD_