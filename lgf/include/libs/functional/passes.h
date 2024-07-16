
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
    virtual resultCode rewrite(painter &p, differentiateOp *op);
  };

  class analyticFuncDerivativeRewriter : public rewriter<partialDifferentiateOp>
  {
  public:
    analyticFuncDerivativeRewriter() = default;
    virtual resultCode rewrite(painter &p, partialDifferentiateOp *op);
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
      add_rewriter<normalizeRewriter>();
      auto result = apply_rewriter_greedy(p, get_graph());
      return result;
    }
  };

  std::unique_ptr<passBase> createCalculusPass();
} // namespace lgf

#endif