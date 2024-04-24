
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
    realInterval(LGFContext *ctx, double l, double r, bool ol = 1, bool or = 1) : setDesc(ctx), dLeftBound(l), dRightBound(r), bLO(ol), bRO(or), bInit(false)
    {
      set_sid("real-interval");
    }
    realInterval(LGFContext *ctx) : setDesc(ctx)
    {
      set_sid("real-interval");
    }

    virtual sid_t represent() override
    {
      auto res = get_sid();
      if (!bInit)
      {
        res += " ";
        std::string lbm = bLO ? "(" : "[";
        std::string rbm = bRO ? ")" : "]";
        res += lbm + utils::to_string(dLeftBound) + ", " + utils::to_string(dRightBound) + rbm;
      }
      return res;
    }

  private:
    const bool bInit = true;
    bool bLO = false, bRO = false;
    double dLeftBound, dRightBound;
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