
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H

#include "lgf/value.h"
#include "lgf/context.h"
#include "lgf/utils.h"

namespace lgf::math
{

  class setDesc : public descImpl
  {
  public:
    setDesc() : descImpl("set") {}
    static descriptor get()
    {
      return descriptor::get<setDesc>();
    }
  };

  class sigmaAlgebra : public descImpl
  {
  public:
    sigmaAlgebra() : descImpl("sigma-algebra") {}
    static descriptor get()
    {
      return descriptor::get<sigmaAlgebra>();
    }
    virtual std::unique_ptr<descImpl> copy()
    {
      return std::make_unique<sigmaAlgebra>();
    }
  };

} // namespace lgf

#endif