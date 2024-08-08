
#ifndef LGF_LIB_ALGEBRA_DESC_H
#define LGF_LIB_ALGEBRA_DESC_H

#include "lgf/attribute.h"
#include "lgf/code.h"
#include "lgf/context.h"
#include "lgf/value.h"
#include "lgf/attribute.h"
#include "lgf/utils.h"

namespace lgf::math
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

  class varDesc : public descImpl
  {
  public:
    varDesc() : descImpl("variable") {}
    inline static descriptor get()
    {
      return descriptor::get<varDesc>();
    }
  };

  class integer : public descImpl
  {
  public:
    integer() : descImpl("integer") {}
    static descriptor get()
    {
      return descriptor::get<integer>();
    }
  };

  // class integerData : public attrBase
  // {
  // public:
  //   integerData(int64_t d) : attrBase("integerData"), data(d) {}
  //   static attribute get(int d = 0)
  //   {
  //     return attribute::get<integerData>(d);
  //   }

  // private:
  //   int data = 0;
  // };

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

  class realNumber : public descImpl
  {
  public:
    realNumber() : descImpl("realNumber"){};
    static descriptor get()
    {
      return descriptor::get<realNumber>();
    }
  };

  class matrix : public descImpl
  {
  public:
    matrix() : descImpl("matrix") {}
    matrix(uint64_t nc, uint64_t nr, descriptor base) : descImpl("matrix"), col(nc), row(nr), elemDesc(base) {}
    matrix(descriptor base) : descImpl("matrix"), elemDesc(base) {}
    static descriptor get(descriptor base)
    {
      return descriptor::get<matrix>(base);
    }
    static descriptor get(uint64_t nc, uint64_t nr, descriptor base)
    {
      return descriptor::get<matrix>(nc, nr, base);
    }
    static descriptor get()
    {
      return descriptor::get<matrix>();
    }
    uint64_t get_dim(int i)
    {
      if (i == 0)
        return col;
      return row;
    }
    descriptor get_elem_desc()
    {
      return elemDesc;
    }

  private:
    descriptor elemDesc;
    uint64_t col = 0, row = 0;
  };

} // namespace lgf

#endif