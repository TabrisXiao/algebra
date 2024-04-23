
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H

#include "lgf/value.h"
#include "lgf/context.h"
#include "lgf/utils.h"

namespace lgf
{

  class setDesc : public simpleValue
  {
  public:
    setDesc(LGFContext *ctx) : simpleValue("set") {}
  };

  class realInterval : public setDesc
  {
  public:
    realInterval(LGFContext *ctx, node *left, node *right, bool ol = 1, bool or = 1) : setDesc(ctx), lb(left), rb(right)
    {
      set_sid("real-interval");
    }
    bool bLO = false, bRO = false;
    node *lb = nullptr, *rb = nullptr;
    virtual sid_t represent() override
    {
      auto res = get_sid() + " ";
      std::string lbm = bLO ? "(" : "[";
      std::string rbm = bRO ? ")" : "]";
      res += lbm + lb->get_value().get_sid() + ", " + rb->get_value().get_sid() + rbm;
      return res;
    }
  };

  class emptySet : public simpleValue
  {
  public:
    emptySet(LGFContext *ctx) : simpleValue("empty-set"){};
  };

  class sigmaAlgebra : public simpleValue
  {
  public:
    sigmaAlgebra(LGFContext *ctx) : simpleValue("sigma-algebra") {}
  };

} // namespace lgf

#endif