
#ifndef LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#define LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#include "lgf/pass.h"
#include "libs/fa/ops.h"
namespace lgf
{
using namespace AAB;
class ChainRuleRewriter: public rewriter<differentiateOp>{
    public:
    ChainRuleRewriter() = default;
    virtual resultCode rewrite(painter p, differentiateOp *op){
        // this apply the differentiation chain rule to all
        // differentiateOp in the graph
        auto result = resultCode::pass();
        value* funcValue = op->input();
        auto ctx = p.getContext();
        operation* funcOp = funcValue->getDefiningOp();
        if(auto assignop = dynamic_cast<assignOp*>(funcOp)){
            funcValue = assignop->rhs();
            funcOp = funcValue->getDefiningOp();
        }
        auto target = op->target();
        if( funcValue != target ){
            auto func = funcValue->getDefiningOp<mappingOp>();
            if(func == nullptr){
                p.setPaintPointBefore(target->getDefiningOp());
                auto zero = p.paint<declOp>(ctx->getType<zeroType>(target->getType()));
                op->replaceBy(zero);
                result.add(resultCode::success());
                return result;
            }
            if(func->getArgNumber()>1){
                // if the input is a mappingOp with more than one argument
                // then we need to apply the chain rule to each argument
                // and sum them up
                p.setPaintPointAfter(op);
                auto sumop = p.paintNoAppend<sumOp>(op->inputValue(0)->getType());
                p.setPaintPointBefore(sumop);
                auto inputs = func->getArugments();
                //p.setPaintPointBefore(op);
                for(size_t i=0; i<inputs.size(); i++){
                    auto intermediateValue = func->argument(i);
                    auto diff1op = p.paint<partialDifferentiateOp>(func->output(), intermediateValue);
                    auto diff2op = p.paint<differentiateOp>(intermediateValue, target);
                    auto prodop = p.paint<productOp>(diff1op->output(), diff2op->output());
                    sumop->addArgument(prodop->output());
                }
                //func->replaceBy(sumop);
                op->replaceBy(sumop);
                result.add(resultCode::success());
                return result;
            } else {
                // if the input is a mappingOp with only one argument
                // then we can apply the chain rule directly
                p.setPaintPointAfter(op);
                auto diff1op = p.paint<partialDifferentiateOp>(func->output(), func->argument(0));
                auto diff2op = p.paint<differentiateOp>(func->argument(0), target);
                auto prodop = p.paint<productOp>(diff1op->output(), diff2op->output());
                op->replaceBy(prodop);
                
                result.add(resultCode::success());
                return result;
            }
        }else {
            p.setPaintPointAfter(target->getDefiningOp());
            auto unit = p.paint<declOp>(ctx->getType<unitType>(target->getType()));
            op->replaceBy(unit);
            result.add(resultCode::success());
        }
        return result;
    }
};

class CalculusPass : public passBase{
    public:
    CalculusPass()
    : passBase("CalculusPass") {}
    virtual resultCode run() final{
        painter p(getContext());
        addRewriter<ChainRuleRewriter>(); 
        auto result = applyRewriterGreedy(p, getGraph());
        return result;
    }
};


std::unique_ptr<passBase> createCalculusPass(){
    return std::make_unique<CalculusPass>();
}
} // namespace lgf


#endif