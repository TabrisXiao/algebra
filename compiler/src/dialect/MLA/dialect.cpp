
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

#include "dialect/MLA/type.hpp"
#include "dialect/MLA/dialect.hpp"
#include "dialect/MLA/op.hpp"

#include "dialect/MLA/generated/dialect.cpp.inc"

void MC::MLA::MLADialect::initialize(){
    registerTypes();
    addOperations<
        #define GET_OP_LIST
        #include "dialect/MLA/generated/op.cpp.inc"
    >();
}

/*
mlir::Type MC::MLA::MLADialect::parseType(mlir::DialectAsmParser &parser) const {
    // Parse: `xTensorBasis` `<`
    if (parser.parseKeyword("xTensorBasis") || parser.parseLess())
        return mlir::Type();

    // Parse the element types of the xTensorBasis.
    mlir::SmallVector<mlir::Type, 1> elementTypes;
    do {
        // Parse the current element type.
        mlir::SMLoc typeLoc = parser.getCurrentLocation();
        mlir::Type elementType;
        if (parser.parseType(elementType)) return nullptr;

        // Check that the type is a DenseI64ArrayAttr.
        if (!elementType.isa<mlir::IntegerType>()) {
        parser.emitError(typeLoc, "element type for a xTensorBasis must be a IntegerType, got: ")
            << elementType;
        return mlir::Type();
        }
        elementTypes.push_back(elementType);
        // Parse the optional: `,`
        } while (succeeded(parser.parseOptionalComma()));

        // Parse: `>`
        if (parser.parseGreater())
            return mlir::Type();
        return xTensorBasisType::get(elementTypes);
}
*/

#define GET_OP_CLASSES
#include "dialect/MLA/generated/op.cpp.inc"