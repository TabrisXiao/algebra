
#include "dialect/AA/AATypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

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
    llvm::StringRef identifier;
    if(parser.parseLess()) return Type();
    if (succeeded(parser.parseKeyword(&identifier))){
        if (!identifier.consume_front("@")) {
            parser.emitError(parser.getCurrentLocation(), "illegal symbolic prefix");
            return nullptr;
        }
        auto ctx = parser.getBuilder().getContext();
        type = MC::AA::AElemType::get(ctx, mlir::FlatSymbolRefAttr::get(ctx, identifier));
    }
    if(parser.parseGreater()) return Type();
    return type;
}

void  MC::AA::AElemType::print(::mlir::AsmPrinter &odsPrinter) const
{
    odsPrinter << "<@";
    odsPrinter<<getEncoding();
    // llvm::StringRef id = getEncoding().dyn_cast<mlir::FlatSymbolRefAttr>().getLeafReference().getValue();
    // odsPrinter << id;
    odsPrinter << ">";
}

#define GET_TYPEDEF_CLASSES
#include "dialect/AA/AATypes.cpp.inc"