
#include "dialect/AA/AATypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

using namespace mlir;
void MC::AA::AADialect::registerTypes() {
    addTypes<
        #define GET_TYPEDEF_LIST
        #include "dialect/AA/AATypes.cpp.inc"
    >();
}

::mlir::Type MC::AA::AElemType::parse(::mlir::AsmParser & parser)
{
    MC::AA::AElemType type;
    NamedAttrList attrList;
    StringAttr nameId;
    if (succeeded(parser.parseOptionalKeyword(getMnemonic().data()))){
        if(failed(parser.parseLess())){
            parser.emitError(parser.getCurrentLocation(), "expected `<`");
            return nullptr;
        }
        if (succeeded(parser.parseOptionalKeyword("Symbol"))){
            if(failed(parser.parseColon())){
                parser.emitError(parser.getCurrentLocation(), "expected `:`");
                return nullptr;
            }
            if(succeeded(parser.parseSymbolName(nameId,"Symbol", attrList))){
                auto ctx = parser.getBuilder().getContext();
                type = MC::AA::AElemType::get(ctx, mlir::FlatSymbolRefAttr::get(ctx, nameId));
            }
        }
        if(failed(parser.parseGreater())){
            parser.emitError(parser.getCurrentLocation(), "expected `>`");
            return nullptr;
        }
    }
    else {
        return nullptr;
    }
    return type;
}

void  MC::AA::AElemType::print(::mlir::AsmPrinter &odsPrinter) const
{
    odsPrinter << "AElement <Symbol: ";
    odsPrinter<<getEncoding();
    odsPrinter << ">";
}

#define GET_TYPEDEF_CLASSES
#include "dialect/AA/AATypes.cpp.inc"