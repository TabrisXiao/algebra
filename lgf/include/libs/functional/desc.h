
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
    realInterval(LGFContext *ctx, double left, double right, bool o1, bool o2)
        : lb(left), rb(right), lop(o1), rop(o2), setDesc(ctx)
    {
      set_sid("real-interval");
    }
    virtual sid_t represent() override
    {
      auto res = get_sid() + " ";
      std::string lbm = lop ? "(" : "[";
      std::string rbm = rop ? ")" : "]";
      res += lbm + lgf::utils::to_string(lb) + ", " + lgf::utils::to_string(rb) +
             rbm;
      return res;
    }
    bool is_belong(double x) const
    {
      if (x > rb)
        return false;
      if (x < lb)
        return false;
      if (x == rb && rop)
        return false;
      if (x == lb && lop)
        return false;
      return true;
    }

  private:
    double lb, rb;
    bool lop, rop;
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

  class funcDesc : public valueDesc
  {
  public:
    funcDesc(LGFContext *ctx, setDesc *d, setDesc *v) : valueDesc("setMeasure") {}
    setDesc *domain = nullptr, *region = nullptr;
  };

  class measureFuncDecs : public funcDesc
  {
  public:
    measureFuncDecs(LGFContext *ctx, setDesc *d) : funcDesc(ctx, d, ctx->get_desc<realInterval>(0, 1, 0, 0)){};
  };

  class probSpace : public valueDesc
  {
  public:
    probSpace(LGFContext *ctx, setDesc *s, sigmaAlgebra *a) : valueDesc("probSpace") {}
  };
} // namespace lgf

#endif