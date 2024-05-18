
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

  class realNumberData : public attrBase
  {
  public:
    enum numberType : int8_t
    {
      real,
      inf,
      ninf,
      e,
      pi
    };

    realNumberData(numberType t, double d = 0) : attrBase("realNumberData"), type(t), data(d) {}
    virtual std::unique_ptr<attrBase> copy() override
    {
      return std::make_unique<realNumberData>(type, data);
    }

    static attribute get(numberType t, double d = 0)
    {
      return attribute::get<realNumberData>(t, d);
    }

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

      throw std::runtime_error("real number data: Unknown data type!");
      return "finite";
    }
    double get_data() { return data; }
    numberType get_type() { return type; }
    static realNumberData get_pinf()
    {
      return realNumberData(realNumberData::inf, 0);
    }
    static realNumberData get_ninf()
    {
      return realNumberData(ninf, 0);
    }
    bool is_unit()
    {
      if (type == real)
        return data == 1;
      return false;
    }
    bool is_zero()
    {
      if (type == real)
      {
        return data == 0;
      }
      return false;
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