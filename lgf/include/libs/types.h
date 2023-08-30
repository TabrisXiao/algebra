#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "lgf/types.h"
#include "lgf/typeTable.h"

namespace lgf::math{

class integer: public lgf::variable {
    public:
    integer() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::typeImpl>("integer"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<integer>();
    }
};

class natureNumber: public integer {
    public:
    natureNumber() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::typeImpl>("natureNumber"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<natureNumber>();
    }
};

class realNumber: public lgf::variable {
    public:
    realNumber() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::typeImpl>("realNumber"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<realNumber>();
    }
};

class rationalNumber: public lgf::variable {
    public:
    rationalNumber() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::typeImpl>("rationalNumber"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<rationalNumber>();
    }
};

class irrationalNumber: public lgf::variable {
  public:
  irrationalNumber() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(){
    return std::move(std::make_unique<lgf::typeImpl>("irrationalNumber"));
  }
  static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
    return ctx->getType<irrationalNumber>();
  }
};

// void registerTypes(){
//   REGISTER_TYPE(integer, "integer");
//   REGISTER_TYPE(natureNumber, "natureNumber");
//   REGISTER_TYPE(realNumber, "realNumber");
//   REGISTER_TYPE(rationalNumber, "rationalNumber");
//   REGISTER_TYPE(irrationalNumber, "irrationalNumber");
//   REGISTER_TYPE(tensor_t, "tensor");
// }

}
#endif
