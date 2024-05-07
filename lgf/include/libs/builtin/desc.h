
#ifndef LGF_BUILTIN_TYPES_H_
#define LGF_BUILTIN_TYPES_H_
#include "lgf/value.h"
#include "lgf/attribute.h"
#include "lgf/utils.h"
#include "lgf/context.h"
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

  class int32Value : public descBase
  {
  public:
    int32Value() : descBase("int32") {}
    int32Value(LGFContext *ctx) : descBase("int32") {}
    static descriptor get()
    {
      return descriptor(std::make_unique<int32Value>());
    }
  };

  class float32Value : public descBase
  {
  public:
    float32Value() : descBase("float32") {}
    float32Value(LGFContext *ctx) : descBase("float32") {}
    static descriptor get()
    {
      return descriptor::get<float32Value>();
    }
  };

  class listDesc : public descBase
  {
  public:
    listDesc() : descBase("list") {}
    template <typename... ARGS>
    listDesc(ARGS... args) : descBase("list")
    {
      auto nds = std::initializer_list<descriptor>{args...};
      for (auto nd : nds)
      {
        data.push_back(nd);
      }
    }
    template <typename... ARGS>
    static descriptor get(ARGS... arg)
    {
      return descriptor::get<listDesc>(arg...);
    }
    virtual std::unique_ptr<descBase> copy() override
    {
      return std::make_unique<listDesc>(*this);
    }
    virtual sid_t represent() override
    {
      sid_t res = get_sid() + "{";
      for (auto desc : data)
      {
        res += desc.get_sid() + ", ";
      }
      res.pop_back();
      res += "}";
      return res;
    }
    size_t size()
    {
      return data.size();
    }
    descriptor at(int i)
    {
      return data[i];
    }

  private:
    std::vector<descriptor> data;
  };

  class funcDesc : public descBase
  {
  public:
    funcDesc() : descBase("func") {}

    funcDesc(descriptor out, std::vector<descriptor> &in) : descBase("func")
    {
      ret = out;
      args = in;
    }
    virtual std::unique_ptr<descBase> copy() override
    {
      return std::make_unique<funcDesc>(ret, args);
    }

    virtual sid_t represent() override
    {
      sid_t res = get_sid() + ": " + "(";
      for (auto desc : args)
      {
        res += desc.get_sid() + ", ";
      }
      res.pop_back();
      res.pop_back();
      res += ")";
      return res;
    }

    descriptor get_arg_desc(int i)
    {
      return args[i];
    }
    descriptor get_ret_desc()
    {
      return ret;
    }
    std::vector<descriptor> &get_arg_descs()
    {
      return args;
    }
    std::vector<descriptor> args;
    descriptor ret;
  };

} // namespace lgf

#endif