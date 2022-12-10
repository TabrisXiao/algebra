
#include "dialect/AA/AADialect.h"
#include "dialect/AA/AAOps.h"
#include "dialect/AA/AATypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

LogicalResult MC::AA::AElemDecl::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    const auto encoding = attributes.get("encoding");
    auto type = MC::AA::AElemType::get(ctx, encoding);
    inferredReturnTypes.push_back(type);
    return mlir::success();
}

LogicalResult MC::AA::Multiply::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    auto type = operands[0].getType();
    inferredReturnTypes.push_back(type);
    return mlir::success();
}

LogicalResult MC::AA::Add::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    auto type = operands[0].getType();
    inferredReturnTypes.push_back(type);
    return mlir::success();
}

LogicalResult MC::AA::Inverse::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    auto type = operands[0].getType();
    inferredReturnTypes.push_back(type);
    return mlir::success();
}

LogicalResult MC::AA::Negative::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    auto type = operands[0].getType();
    inferredReturnTypes.push_back(type);
    return mlir::success();
}