
#ifndef LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#define LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#include "lgf/pass.h"
#include "libs/algebra/desc.h"
#include "ops.h"
namespace lgf {
class ChainRuleRewriter : public rewriter<differentiateOp> {
public:
  ChainRuleRewriter() = default;
  virtual resultCode rewrite(painter p, differentiateOp *op) {
    // this apply the differentiation chain rule to all
    // differentiateOp in the graph
    auto result = resultCode::pass();
    if (op->get_order() > 1) {
      auto d =
          p.paint<differentiateOp>(op->arg(), op->arg(1), op->get_order() - 1);
      op->replace_input_by(op->arg(), d);
      op->set_order(op->get_order() - 1);
      return resultCode::success();
    }

    node *func = op->arg();
    node *target = op->arg(1);
    auto ctx = p.get_context();

    if (func == target) {
      p.replace_op<declOp>(op,
                           ctx->get_desc<unitDesc>(target->get_value_desc()));
      result.add(resultCode::success());
      return result;
    }

    if (auto f = dynamic_cast<declOp *>(func)) {
      p.replace_op<declOp>(op,
                           ctx->get_desc<zeroDesc>(target->get_value_desc()));
      result.add(resultCode::success());
      return result;
    }

    auto mapping = dynamic_cast<mappingOp *>(func);

    // for(auto &h : mapping->get_input_handles()){
    //     if(!h.is_coupled()) continue;
    //     auto arg = h.get_dual_node();
    //     auto partial1 = p.paint<partialDifferentiateOp>( mapping, arg, 1);
    //     auto partial2 = p.paint<partialDifferentiateOp>(arg, target, 1);
    //     auto product = p.replace_op<productOp>(op, partial1, partial2);
    //     result.add(resultCode::success());
    // }

    // if(func->getArgNumber()>1){
    //     // if the input is a mappingOp with more than one argument
    //     // then we need to apply the chain rule to each argument
    //     // and sum them up
    //     p.setPaintPointAfter(op);
    //     auto sumop = p.paintNoAppend<sumOp>(op->inputValue(0)->getType());
    //     p.setPaintPointBefore(sumop);
    //     auto inputs = func->getArugments();
    //     //p.setPaintPointBefore(op);
    //     for(size_t i=0; i<inputs.size(); i++){
    //         auto intermediateValue = func->argument(i);
    //         auto diff1op = p.paint<partialDifferentiateOp>(func->output(),
    //         intermediateValue); auto diff2op =
    //         p.paint<differentiateOp>(intermediateValue, target); auto prodop
    //         = p.paint<productOp>(diff1op->output(), diff2op->output());
    //         sumop->addArgument(prodop->output());
    //     }
    //     //func->replaceBy(sumop);
    //     op->replaceBy(sumop);
    //     result.add(resultCode::success());
    //     return result;
    // } else {
    //     // if the input is a mappingOp with only one argument
    //     // then we can apply the chain rule directly
    //     p.setPaintPointAfter(op);
    //     auto diff1op = p.paint<partialDifferentiateOp>(func->output(),
    //     func->argument(0)); auto diff2op =
    //     p.paint<differentiateOp>(func->argument(0), target); auto prodop =
    //     p.paint<productOp>(diff1op->output(), diff2op->output());
    //     op->replaceBy(prodop);
    //     result.add(resultCode::success());
    //     return result;
    // }
    return result;
  }
};

class analyticFuncDerivativeRewriter : public rewriter<partialDifferentiateOp> {
public:
  analyticFuncDerivativeRewriter() = default;
  virtual resultCode rewrite(painter p, partialDifferentiateOp *op) {
    auto result = resultCode::pass();
    // auto ctx = p.getContext();
    // auto func = op->func()->getDefiningOp<mappingOp>();
    // if( !func ) return result;
    // auto target = op->var();
    // if(auto sum = dynamic_cast<sumOp*>(func)){
    //     p.setPaintPointAfter(op);
    //     sum = p.paint<sumOp>(sum->getArugments());
    //     p.setPaintPointToTop();
    //     for(auto arg : sum->getArugments()){
    //         if(arg == target){
    //             auto one =
    //             p.paint<declOp>(ctx->getType<unitType>(target->getType()));
    //             sum->replaceInputValueBy(arg, one->output());
    //         }
    //     }
    //     p.setPaintPointAfter(op);
    //     op->replaceBy(sum);
    //     result.add(resultCode::success());
    //     return result;
    // }
    // if(auto product = dynamic_cast<productOp*>(func)){
    //     p.setPaintPointAfter(op);
    //     product = p.paint<productOp>(product->getArugments());
    //     p.setPaintPointToTop();
    //     for(auto arg : product->getArugments()){
    //         if(arg == target){
    //             auto one =
    //             p.paint<declOp>(ctx->getType<unitType>(target->getType()));
    //             product->replaceInputValueBy(arg, one->output());
    //         }
    //     }
    //     p.setPaintPointAfter(op);
    //     op->replaceBy(product);
    //     result.add(resultCode::success());
    //     return result;
    // }

    // if( func->getArgNumber() > 1) return result;
    // if(func->argument(0) != target) return result;
    // if(auto sinf = dynamic_cast<funcSineOp*>(func)){
    //     p.setPaintPointAfter(op);
    //     auto cosf = p.paint<funcCosOp>(sinf->argument(0));
    //     op->replaceBy(cosf);
    //     result.add(resultCode::success());
    //     return result;
    // }else if(auto cosf = dynamic_cast<funcCosOp*>(func)){
    //     p.setPaintPointAfter(op);
    //     auto sinf = p.paint<funcSineOp>(cosf->argument(0));
    //     auto neg = p.paint<negativeOp>(sinf->output());
    //     op->replaceBy(neg);
    //     result.add(resultCode::success());
    //     return result;
    // }else if(auto powerf = dynamic_cast<funcPowerOp*>(func)){
    //     p.setPaintPointAfter(op);
    //     auto x = powerf->x();
    //     if(x == target){
    //         auto power = p.paint<funcPowerOp>(x, powerf->power()-1);
    //         op->replaceBy(power);
    //     }else{
    //         auto zero =
    //         p.paint<declOp>(ctx->getType<zeroType>(target->getType()));
    //         op->replaceBy(zero);
    //     }
    //     result.add(resultCode::success());
    //     return result;
    // }
    return result;
  }
};

class CalculusPass : public passBase {
public:
  CalculusPass() : passBase("CalculusPass") {}
  virtual resultCode run() final {
    painter p(get_graph());
    add_rewriter<ChainRuleRewriter>();
    add_rewriter<analyticFuncDerivativeRewriter>();
    auto result = apply_rewriter_greedy(p, get_graph());
    return result;
  }
};

std::unique_ptr<passBase> createCalculusPass() {
  return std::make_unique<CalculusPass>();
}
} // namespace lgf

#endif