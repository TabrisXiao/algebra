
#include "AA/AADialect.h"
#include "AA/AAOps.h"
#include "AA/AATypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

LogicalResult MC::AA::AElemDecl::inferReturnTypes(MLIRContext *ctx, llvm::Optional<Location> loc, ValueRange operands, DictionaryAttr attributes, RegionRange regions, llvm::SmallVectorImpl<Type> &inferredReturnTypes){
    mlir::Attribute encoding = attributes.getNamed("encoding").getValue().getValue();
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

template<typename ConcreteOp>
struct DoubleOccurrence : public mlir::OpRewritePattern<ConcreteOp> {
  /// We register this pattern to match every concrete op in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  DoubleOccurrence(mlir::MLIRContext *context)
      : OpRewritePattern<ConcreteOp>(context, /*benefit=*/1) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(ConcreteOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current Op.
    mlir::Value input = op.getOperand();
    ConcreteOp inputOp = input.getDefiningOp<ConcreteOp>();

    // Input defined by another ConcreteOp? If not, no match.
    if (!inputOp)
      return failure();

    // Otherwise, we have a redundant ConcreteOp. Use the rewriter.
    rewriter.replaceOp(op, {inputOp.getOperand()});
    return success();
  }
};

void MC::AA::Negative::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<DoubleOccurrence<MC::AA::Negative>>(context);
}

void MC::AA::Inverse::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<DoubleOccurrence<MC::AA::Inverse>>(context);
}