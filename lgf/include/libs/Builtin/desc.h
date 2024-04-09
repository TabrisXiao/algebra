
#ifndef LGF_BUILTIN_TYPES_H_
#define LGF_BUILTIN_TYPES_H_
#include "lgf/value.h"
#include "lgf/attribute.h"
namespace lgf
{

  class intData : public preservedDataAttr<int>
  {
  public:
    intData(int d = 0) : preservedDataAttr<int>("int", d) {}
    virtual sid_t represent_data() override
    {
      return std::to_string(get_data());
    }
  };

  class doubleData : public preservedDataAttr<double>
  {
  public:
    doubleData(double d = 0) : preservedDataAttr<double>("double", d) {}
    virtual sid_t represent_data() override
    {
      return std::to_string(get_data());
    }
  };

  class intValue : public simpleValue
  {
  public:
    intValue() : simpleValue("int") {}
  };

  class doubleValue : public simpleValue
  {
  public:
    doubleValue() : simpleValue("double") {}
  };

} // namespace lgf

#endif