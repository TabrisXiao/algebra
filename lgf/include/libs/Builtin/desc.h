
#ifndef LGF_BUILTIN_TYPES_H_
#define LGF_BUILTIN_TYPES_H_
#include "lgf/value.h"
#include "lgf/attribute.h"
#include "lgf/utils.h"
namespace lgf
{
  class LFFContext;
  class int32Data : public preservedDataAttr<int>
  {
  public:
    int32Data(int d = 0) : preservedDataAttr<int>("i32", d) {}
    virtual sid_t represent_data() override
    {
      return utils::to_string(get_data());
    }
  };

  class float32Data : public preservedDataAttr<double>
  {
  public:
    float32Data(double d = 0) : preservedDataAttr<double>("f32", d) {}
    virtual sid_t represent_data() override
    {
      return utils::to_string(get_data());
    }
  };

  class int32Value : public simpleValue
  {
  public:
    int32Value(LGFContext *ctx) : simpleValue("int32") {}
  };

  class float32Value : public simpleValue
  {
  public:
    float32Value(LGFContext *ctx) : simpleValue("float32") {}
  };

} // namespace lgf

#endif