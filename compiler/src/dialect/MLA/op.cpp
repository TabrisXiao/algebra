
#include "dialect/MLA/dialect.hpp"
#include "dialect/MLA/op.hpp"
#include "dialect/MLA/type.hpp"
#include "mlir/IR/Operation.h"

using namespace mlir;
/*
LogicalResult MC::MLA::TensorBasisDecl::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    const auto encoding = attributes.get("encoding");
    auto array = attributes.get("shape").dyn_cast<DenseI32ArrayAttr>();
    auto type = MC::MLA::xTensorBasisType::get(ctx, array, encoding);
    inferredReturnTypes.push_back(type);
    return mlir::success();
}
*/
