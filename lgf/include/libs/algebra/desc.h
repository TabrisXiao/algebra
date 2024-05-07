
#ifndef LGF_LIB_ALGEBRA_DESC_H
#define LGF_LIB_ALGEBRA_DESC_H

#include "lgf/attribute.h"
#include "lgf/code.h"
#include "lgf/context.h"
#include "lgf/value.h"
#include "lgf/attribute.h"
#include "lgf/utils.h"

namespace lgf
{
  // template<typename T>
  // class interfaceBase {
  //   public:
  //   interfaceBase() = default;
  //   virtual ~interfaceBase() = default;
  //   void binary_op(node* lhs, node* rhs){
  //     return T::binary_op(lhs, rhs);
  //   }
  // };

  class varDesc : public descBase
  {
  public:
    varDesc() : descBase("variable") {}
    inline static descriptor get()
    {
      return descriptor::get<varDesc>();
    }
    virtual std::unique_ptr<descBase> copy() override
    {
      return std::make_unique<varDesc>();
    }
  };

  class realNumberAttr : public dataAttr
  {
  public:
    enum numberType : int8_t
    {
      real,
      one,
      inf,
      ninf,
      e,
      pi,
      finite
    };
    realNumberAttr(double d) : dataAttr("realNumbere"), data(d) {}
    realNumberAttr(numberType t) : dataAttr("realNumber"), type(t) {}
    // realNumberAttr(realNumberAttr &&attr) : dataAttr("realNumber"), data(attr.get_data()), type(attr.get_type()) {}
    void set_type(numberType t) { type = t; }

    virtual sid_t represent() override
    {
      if (type == real)
        return utils::to_string(data);
      if (type == inf)
        return "inf";
      if (type == ninf)
        return "-inf";
      if (type == e)
        return "e";
      if (type == pi)
        return "pi";
      return "finite";
    }
    double get_data() { return data; }
    numberType get_type() { return type; }
    static realNumberAttr get_pinf()
    {
      realNumberAttr res(0);
      res.set_type(inf);
      return res;
    }
    static realNumberAttr get_ninf()
    {
      realNumberAttr res(0);
      res.set_type(ninf);
      return res;
    }

  private:
    numberType type = real;
    double data = 0;
  };

  class realNumber : public descBase
  {
  public:
    realNumber() : descBase("realNumber"){};
    static descriptor get()
    {
      return descriptor::get<realNumber>();
    }
    virtual std::unique_ptr<descBase> copy() override { return std::make_unique<realNumber>(); }
  };

} // namespace lgf

#endif