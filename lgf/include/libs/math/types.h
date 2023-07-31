#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "lgf/types.h"

namespace math{

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

class tensorTypeImpl : public lgf::typeImpl {
  public:
  tensorTypeImpl(lgf::type_t elemType_, int rank, int* shapes)
  : typeImpl("tensor")
  , elemType(elemType_){
    THROW_WHEN(rank<1, "tensor rank can't be less than 1!");
    dims.reserve(rank);
    for(auto i=0;i<rank; i++){
      dims.push_back(shapes[i]);
    }
  }
  virtual std::string reprsent(){
    printer p;
    p<<id<<"<"<<dims[0];
    if(dims.size()>1){
      for(auto i=1; i<dims.size(); i++){
        p<<"x"<<dims[i];
      }
    }
    p<<", "<<elemType.represent()<<">";
    return p.dump();
  }
  lgf::type_t elemType;
  std::vector<int> dims;
};

class tensor: public lgf::variable {
  public:
  tensor() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType, int size, int* shapes){
    return std::move(std::make_unique<tensorTypeImpl>(elemType, size, shapes));
  }
  int size(){
    return dynamic_cast<tensorTypeImpl*>(impl)->dims.size();
  }
  type_t getElemType(){ return dynamic_cast<tensorTypeImpl*>(impl)->elemType; }
};

void registerLGFTypes(){
  REGISTER_TYPE(integer, "integer");
  REGISTER_TYPE(natureNumber, "natureNumber");
  REGISTER_TYPE(realNumber, "realNumber");
  REGISTER_TYPE(rationalNumber, "rationalNumber");
  REGISTER_TYPE(irrationalNumber, "irrationalNumber");
  REGISTER_TYPE(tensor, "tensor");
}

}
#endif
