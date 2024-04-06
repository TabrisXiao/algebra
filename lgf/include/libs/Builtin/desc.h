
#ifndef LGF_BUILTIN_TYPES_H_
#define LGF_BUILTIN_TYPES_H_
#include "lgf/value.h"

namespace lgf
{
  class dataAttr
  {
  public:
  dataAttr() = default;
    dataAttr(sid_t id) : sid(id){};
    virtual sid_t represent() = 0;
    sid_t get_sid()
    {
      return sid;
    }

  private:
    sid_t sid;
  };

  // preserved data attribute contains contains the data itself.
  // this is useful when cost of data copy is low and make the
  // attribute creataion more convenient.
  template <typename T>
  class preservedDataAttr : public dataAttr
  {
  public:
    preservedDataAttr(sid_t id, T d) : data(d), dataAttr(id){};
    virtual sid_t represent()
    {
      return get_sid() + ", val = " + represent_data();
    }
    virtual sid_t represent_data() = 0;
    void set_data(T d)
    {
      data = d;
    }
    T get_data()
    {
      return data;
    }

  private:
    T data;
  };

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

  // simple value is the value that can be identified by a single sid.
  template <typename T>
  class simpleValue : public valueDesc
  {
  public:
    simpleValue(sid_t id) : valueDesc(id) {}
    virtual sid_t represent() override
    {
      return get_sid();
    }
  };

  class intValue : public simpleValue<int>
  {
  public:
    intValue() : simpleValue<int>("int") {}
  };

  class doubleValue : public simpleValue<double>
  {
  public:
    doubleValue() : simpleValue<double>("double") {}
  };

} // namespace lgf

#endif