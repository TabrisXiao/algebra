
#include "AA/AATypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

using namespace mlir;
void MC::AA::AADialect::registerTypes() {
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "AA/AATypes.cpp.inc"
    >();
}

::mlir::Type MC::AA::AElemType::parse(::mlir::AsmParser & parser)
{
    //parsing keyword of AElemType is handled by the parser pointed to this function. 
    //This parser only need to handle the string
    //```
    //<Symbol: @stringRef>
    //```
    std::cout<<"parsing AElemType" <<std::endl;
    MC::AA::AElemType type;
    NamedAttrList attrList;
    StringAttr nameId;
    std::cout<<"parsing AElemType keyword" <<std::endl;
    if(failed(parser.parseLess())){
        parser.emitError(parser.getCurrentLocation(), "expected `<`");
        return nullptr;
    }
    std::cout<<"parsing AElemType Symbol" <<std::endl;
    if (succeeded(parser.parseOptionalKeyword("Symbol"))){
        if(failed(parser.parseColon())){
            parser.emitError(parser.getCurrentLocation(), "expected `:`");
            return nullptr;
        }
        std::cout<<"parsing AElemType StringRef" <<std::endl;
        if(succeeded(parser.parseSymbolName(nameId,"Symbol", attrList))){
            auto ctx = parser.getBuilder().getContext();
            std::cout<<"successed parsed AElemType" <<std::endl;
            type = MC::AA::AElemType::get(ctx, mlir::FlatSymbolRefAttr::get(ctx, nameId));
        }
    }
    if(failed(parser.parseGreater())){
        parser.emitError(parser.getCurrentLocation(), "expected `>`");
        return nullptr;
    }
    return type;
}

void  MC::AA::AElemType::print(::mlir::AsmPrinter &odsPrinter) const
{
    odsPrinter << "<Symbol: ";
    odsPrinter<<getEncoding();
    odsPrinter << ">";
}

#define GET_TYPEDEF_CLASSES
#include "AA/AATypes.cpp.inc"