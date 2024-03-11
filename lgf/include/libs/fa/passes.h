
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
        auto target = op->target();
        auto ctx = p.getContext();
        if( funcValue == target ){
            p.setPaintPointAfter(target->getDefiningOp());
            auto unit = p.paint<declOp>(ctx->getType<unitType>(target->getType()));
            op->replaceBy(unit);
            result.add(resultCode::success());
            return result;
        }
        auto func = funcValue->getDefiningOp<mappingOp>();
        if( func == nullptr ) {
            p.setPaintPointAfter(target->getDefiningOp());
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
        return result;
    }
};

class analyticFuncDerivativeRewriter : public rewriter<partialDifferentiateOp>{
    public:
    analyticFuncDerivativeRewriter() = default;
    virtual resultCode rewrite(painter p, partialDifferentiateOp *op){
        auto result = resultCode::pass();
        auto ctx = p.getContext();
        auto func = op->func()->getDefiningOp<mappingOp>();
        if( !func ) return result;
        auto target = op->var();
        if(auto sum = dynamic_cast<AAB::sumOp*>(func)){
            p.setPaintPointAfter(op);
            sum = p.paint<sumOp>(sum->getArugments());
            p.setPaintPointToTop();
            for(auto arg : sum->getArugments()){
                if(arg == target){
                    auto one = p.paint<declOp>(ctx->getType<unitType>(target->getType()));
                    sum->replaceInputValueBy(arg, one->output());
                }
            }
            p.setPaintPointAfter(op);
            op->replaceBy(sum);
            result.add(resultCode::success());
            return result;
        }
        if(auto product = dynamic_cast<AAB::productOp*>(func)){
            p.setPaintPointAfter(op);
            product = p.paint<productOp>(product->getArugments());
            p.setPaintPointToTop();
            for(auto arg : product->getArugments()){
                if(arg == target){
                    auto one = p.paint<declOp>(ctx->getType<unitType>(target->getType()));
                    product->replaceInputValueBy(arg, one->output());
                }
            }
            p.setPaintPointAfter(op);
            op->replaceBy(product);
            result.add(resultCode::success());
            return result;
        }

        if( func->getArgNumber() > 1) return result;
        if(func->argument(0) != target) return result;
        if(auto sinf = dynamic_cast<funcSineOp*>(func)){
            p.setPaintPointAfter(op);
            auto cosf = p.paint<funcCosOp>(sinf->argument(0));
            op->replaceBy(cosf);
            result.add(resultCode::success());
            return result;
        }else if(auto cosf = dynamic_cast<funcCosOp*>(func)){
            p.setPaintPointAfter(op);
            auto sinf = p.paint<funcSineOp>(cosf->argument(0));
            auto neg = p.paint<negativeOp>(sinf->output());
            op->replaceBy(neg);
            result.add(resultCode::success());
            return result;
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
        addRewriter<analyticFuncDerivativeRewriter>(); 
        auto result = applyRewriterGreedy(p, getGraph());
        return result;
    }
};


std::unique_ptr<passBase> createCalculusPass(){
    return std::make_unique<CalculusPass>();
}
} // namespace lgf


#endif