
#ifndef LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#define LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#include "lgf/pass.h"
#include "libs/algebra/algebra.h"
#include "ops.h"
namespace lgf
{
  class ChainRuleRewriter : public rewriter<differentiateOp>
  {
  public:
    ChainRuleRewriter() = default;
    virtual resultCode rewrite(painter &p, differentiateOp *op)
    {
      // this apply the differentiation chain rule to all
      // differentiateOp in the graph
      auto result = resultCode::pass();

      node *func = op->arg();
      auto ctx = p.get_context();

      if (auto f = dynamic_cast<declOp *>(func))
      {
        return result;
      }

      if (dynamic_cast<differentiateOp *>(func) ||
          dynamic_cast<partialDifferentiateOp *>(func))
      {
        return resultCode::pass();
      }

      if (auto sum = dynamic_cast<sumOp *>(func))
      {
        for (auto &h : func->get_input_handles())
        {
          if (!h.is_coupled())
            continue;
          auto arg = h.get_dual_node();
          auto df = p.paint<differentiateOp>(arg);
          sum->replace_input_by(h, df);
        }
        return resultCode::success();
      }

      if (auto product = dynamic_cast<productOp *>(func))
      {
        std::vector<node *> sum_args;
        for (auto &h : func->get_input_handles())
        {
          auto arg = h.get_dual_node();
          auto df = p.paint<differentiateOp>(arg);
          auto prod = p.paint<productOp>();
          for (auto &h1 : func->get_input_handles())
          {
            if (&h == &h1)
            {
              prod->register_input(df);
            }
            else
            {
              prod->register_input(h1.get_dual_node());
            }
          }
          prod->infer_trivial_value_desc();
          sum_args.push_back(prod);
        }
        p.replace_op<sumOp>(op, sum_args);
        return resultCode::success();
      }

      if (auto mapping = dynamic_cast<mappingOp *>(func))
      {
        std::vector<node *> sum_args;
        for (auto &h : func->get_input_handles())
        {
          auto arg = h.get_dual_node();
          auto Df = p.paint<partialDifferentiateOp>(func, arg);
          auto dx = p.paint<differentiateOp>(arg);
          auto product = p.paint<productOp>(Df, dx);
          sum_args.push_back(product);
        }
        if (sum_args.size() == 1)
        {
          p.replace_op(op, sum_args[0]);
        }
        else
          p.replace_op<sumOp>(op, sum_args);
        result.add(resultCode::success());
      }

      return result;
    }
  };

  class analyticFuncDerivativeRewriter : public rewriter<partialDifferentiateOp>
  {
  public:
    analyticFuncDerivativeRewriter() = default;
    virtual resultCode rewrite(painter &p, partialDifferentiateOp *op)
    {
      auto result = resultCode::pass();

      auto func = op->func();
      auto target = op->var();
      if (func == target)
      {
        p.replace_op<declOp>(
            op, p.get_context()->get_desc<unitDesc>(target->get_value_desc()));
        result.add(resultCode::success());
        return result;
      }

      if (auto sum = dynamic_cast<sumOp *>(func))
      {
        for (auto &h : func->get_input_handles())
        {
          if (!h.is_coupled())
            continue;
          auto arg = h.get_dual_node();
          auto df = p.paint<partialDifferentiateOp>(arg, target);
          sum->replace_input_by(h, df);
        }
        return resultCode::success();
      }

      if (auto product = dynamic_cast<productOp *>(func))
      {
        std::vector<node *> sum_args;
        std::cout << func->represent() << std::endl;
        for (auto &h : func->get_input_handles())
        {
          auto arg = h.get_dual_node();
          auto df = p.paint<partialDifferentiateOp>(arg, target);
          auto prod = p.paint<productOp>();
          for (auto &h1 : func->get_input_handles())
          {
            if (&h == &h1)
            {
              prod->register_input(df);
            }
            else
            {
              prod->register_input(arg);
            }
          }
          prod->infer_trivial_value_desc();
          sum_args.push_back(prod);
        }
        p.replace_op<sumOp>(op, sum_args);

        return resultCode::success();
      }

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

  class CalculusPass : public passBase
  {
  public:
    CalculusPass() : passBase("CalculusPass") {}
    virtual resultCode run() final
    {
      painter p(get_graph());
      add_rewriter<ChainRuleRewriter>();
      add_rewriter<analyticFuncDerivativeRewriter>();
      auto result = apply_rewriter_greedy(p, get_graph());
      return result;
    }
  };

  std::unique_ptr<passBase> createCalculusPass()
  {
    return std::make_unique<CalculusPass>();
  }
} // namespace lgf

#endif