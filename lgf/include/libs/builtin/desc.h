
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

  class listDesc : public valueDesc
  {
  public:
    listDesc(LGFContext *ctx) : valueDesc("list") {}
    template <typename... ARGS>
    listDesc(LGFContext *ctx, ARGS... args) : valueDesc("list")
    {
      auto nds = std::initializer_list<valueDesc *>{args...};
      for (auto nd : nds)
      {
        data.push_back(nd);
      }
    }
    virtual sid_t represent() override
    {
      sid_t res = get_sid() + "{";
      for (auto desc : data)
      {
        res += desc->get_sid() + ", ";
      }
      res.pop_back();
      res += "}";
      return res;
    }
    size_t size()
    {
      return data.size();
    }
    valueDesc *get(int i)
    {
      return data[i];
    }

  private:
    std::vector<valueDesc *> data;
  };

  class funcDesc : public valueDesc
  {
  public:
    funcDesc(LGFContext *ctx) : valueDesc("func") {}

    funcDesc(LGFContext *ctx, valueDesc *out, std::vector<valueDesc *> &in) : valueDesc("func")
    {
      ret = out;
      args = in;
    }

    virtual sid_t represent() override
    {
      sid_t res = get_sid() + ": " + "(";
      for (auto desc : args)
      {
        res += desc->get_sid() + ", ";
      }
      res.pop_back();
      res.pop_back();
      res += ")";
      return res;
    }

    valueDesc *get_arg_desc(int i)
    {
      return args[i];
    }
    valueDesc *get_ret_desc()
    {
      return ret;
    }
    std::vector<valueDesc *> &get_arg_descs()
    {
      return args;
    }
    std::vector<valueDesc *> args;
    valueDesc *ret = nullptr;
  };

} // namespace lgf

#endif