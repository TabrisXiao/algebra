
#ifndef LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#define LGF_FUNCTIONAL_ANALYSIS_PASSES_H
#include "lgf/pass.h"
#include "normalization.h"
#include "ops.h"
namespace lgf::math
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

  class ReimannIntegralRewriter : public rewriter<RiemannIntegralOp>
  {
  public:
    ReimannIntegralRewriter() = default;
    virtual resultCode rewrite(painter &p, RiemannIntegralOp *op) override;
  };

  class CalculusPass : public passBase
  {
  public:
    CalculusPass() : passBase("CalculusPass") {}
    virtual resultCode run() final
    {
      painter p(get_region());
      add_rewriter<ReimannIntegralRewriter>();
      add_rewriter<ChainRuleRewriter>();
      add_rewriter<analyticFuncDerivativeRewriter>();
      add_rewriter<unitRewriter>();
      add_rewriter<zeroRewriter>();
      add_rewriter<normalizeRewriter>();
      auto result = apply_rewriter_greedy(p, get_region());
      return result;
    }
  };

  std::unique_ptr<passBase> createCalculusPass();
} // namespace lgf

#endif