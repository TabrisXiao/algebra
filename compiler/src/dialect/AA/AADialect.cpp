
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "dialect/AA/AATypes.h"
#include "dialect/AA/AADialect.h"
#include "dialect/AA/AAOps.h"

#include "dialect/AA/AADialect.cpp.inc"
using namespace mlir;
void MC::AA::AADialect::initialize(){
    registerTypes();
    addOperations<
        #define GET_OP_LIST
        #include "dialect/AA/AAOps.cpp.inc"
    >();
}
/*
static Type parseAElemType(::mlir::DialectAsmParser & parser)
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

mlir::Type MC::AA::AADialect::parseType(::mlir::DialectAsmParser &parser) const
{
    StringRef typeNameSpelling;
    if (failed(parser.parseKeyword(&typeNameSpelling))) return nullptr;

    if(typeNameSpelling == "AElemType")
            return parseAElemType(parser);
    else {
        parser.emitError(parser.getNameLoc(), "Unknown type `"+typeNameSpelling+"` in AADialect!");
    }
    return nullptr;
}

static void printAElemType(MC::AA::AElemType type, DialectAsmPrinter &out) {
    out << "AElemType<@";
    llvm::StringRef id = type.getEncoding().dyn_cast<mlir::FlatSymbolRefAttr>().getLeafReference().getValue();
    out << id;
    out << ">";
}

void MC::AA::AADialect::printType(::mlir::Type type,  ::mlir::DialectAsmPrinter &os) const 
{
    if (auto anyType = type.dyn_cast<MC::AA::AElemType>())
        printAElemType(anyType, os);
    else
    llvm_unreachable("Unhandled symbolic type");
}


*/

#define GET_OP_CLASSES
#include "dialect/AA/AAOps.cpp.inc"