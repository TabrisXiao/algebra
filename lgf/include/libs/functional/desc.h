
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H

#include "lgf/value.h"
#include "lgf/context.h"
#include "lgf/utils.h"

namespace lgf
{

  class setDesc : public descBase
  {
  public:
    setDesc() : descBase("set") {}
    static descriptor get()
    {
      return descriptor::get<setDesc>();
    }
    virtual std::unique_ptr<descBase> copy() override
    {
      return std::make_unique<setDesc>();
    }
  };

  class sigmaAlgebra : public descBase
  {
  public:
    sigmaAlgebra() : descBase("sigma-algebra") {}
    static descriptor get()
    {
      return descriptor::get<sigmaAlgebra>();
    }
    virtual std::unique_ptr<descBase> copy()
    {
      return std::make_unique<sigmaAlgebra>();
    }
  };

} // namespace lgf

#endif