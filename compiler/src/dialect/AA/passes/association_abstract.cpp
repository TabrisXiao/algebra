
#include <iostream>
#include <unordered_map>
#include "AA/AADialect.h"
#include "AA/AAOps.h"
#include "AA/AATypes.h"
#include "AA/AAPasses.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace MC
{
namespace AA
{

struct AssociationAbstractPass : public AssociationAbstractBase<AssociationAbstractPass>
{
    public:
    struct AssociationAbstractRewriter : public mlir::OpRewritePattern<mlir::func::FuncOp> {
        AssociationAbstractRewriter(mlir::MLIRContext *ctx) : OpRewritePattern<mlir::func::FuncOp>(ctx) {}
        // ---------------------------------------------
        void scanFactor(mlir::Operation *op){
            if(auto mulop = mlir::dyn_cast<MC::AA::Multiply>(*op)){
                registValue(mulop.lhs(), mulop, lhs_map);
                registValue(mulop.rhs(), mulop, rhs_map);
            }
        }
        // ---------------------------------------------
        void registValue(mlir::Value value, 
            MC::AA::Multiply op, std::unordered_map<mlir::Value *, std::vector<MC::AA::Multiply*>> & map){
            if(map.find(&value)==map.end())
                map.insert(std::make_pair<mlir::Value *, std::vector<MC::AA::Multiply*>>(&value, std::vector<MC::AA::Multiply*>()));
            map[&value].push_back(&op);
        }
        // ---------------------------------------------
        mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp mop, mlir::PatternRewriter &rewriter) const override 
        {
            bool isExhausted=1;
            const auto callback = [&](mlir::Operation *op){
                // scan the lhs/rhs of add operations and register all this add op 
                // if their inputs come from multiply op.
                std::cout<<"walking through op: "<<op->getName().getStringRef().data()<<std::endl;
                // if(auto add = mlir::dyn_cast<MC::AA::Add>(*op)){
                //     // scan the lhs input to record all the inputs from multiply op
                //     scanFactor(add.lhs().getDefiningOp());
                //     // do the same thing for rhs
                //     scanFactor(add.rhs().getDefiningOp());
                // }
                return; 
            };
            mop.walk(callback);
            if(isExhausted) mlir::failure();
            return mlir::success();
        }
        std::unordered_map<mlir::Value *, std::vector<MC::AA::Multiply*>> lhs_map;
        std::unordered_map<mlir::Value *, std::vector<MC::AA::Multiply*>> rhs_map;
    };

    
    void runOnOperation() override {
        mlir::MLIRContext *  ctx = &getContext();
        mlir::RewritePatternSet patterns(ctx);
        const auto mop = getOperation();
        patterns.add<AssociationAbstractRewriter>(ctx);
        if(mlir::failed(applyPatternsAndFoldGreedily(mop, std::move(patterns)))){
            signalPassFailure();
        }
    }
};

}
}

std::unique_ptr<mlir::Pass> MC::AA::createAssociationAbstractPass(){
    return std::make_unique<AssociationAbstractPass>();
}