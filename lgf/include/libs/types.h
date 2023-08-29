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

class tensorTypeImpl : public lgf::typeImpl {
  public:
  tensorTypeImpl(lgf::type_t elemType_, std::vector<int>& shapes)
  : typeImpl("tensor")
  , elemType(elemType_){
    THROW_WHEN(shapes.size()<1, "tensor rank can't be less than 1!");
    dims = shapes;
  }
  tensorTypeImpl(lgf::type_t elemType_)
  : typeImpl("tensor")
  , elemType(elemType_){
    generalType = 1;
  }
  virtual std::string represent(){
    printer p;
    p<<id<<"<";
    if(!generalType){
      p<<dims[0];
      if(dims.size()>1){
        for(auto i=1; i<dims.size(); i++){
          p<<"x"<<dims[i];
        }
      }
      p<<", ";
    }
    p<<elemType.represent()<<">";
    return p.dump();
  }
  lgf::type_t elemType;
  std::vector<int> dims;
  bool generalType = 0;
};

class tensor_t: public lgf::variable {
  public:
  tensor_t() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType, std::vector<int> &shapes){
    return std::move(std::make_unique<tensorTypeImpl>(elemType, shapes));
  }
  static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType){
    return std::move(std::make_unique<tensorTypeImpl>(elemType));
  }
  int size(){
    return dynamic_cast<tensorTypeImpl*>(impl)->dims.size();
  }
  std::vector<int>& shape(){
    return dynamic_cast<tensorTypeImpl*>(impl)->dims;
  }
  bool isAbstract () { return dynamic_cast<tensorTypeImpl*>(impl)->generalType; }
  type_t getElemType(){ return dynamic_cast<tensorTypeImpl*>(impl)->elemType; }
  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    p.parseLessThan();
    bool isgeneral = 1;
    std::vector<int> shapes;
    if(p.getCurToken() == liteParser::tok_number){
      int n = p.parseNumber();
      shapes.push_back(n);
      while(p.getCurToken()!=int(',')){
        p.consume(int('x'));
        n = p.parseNumber();
        shapes.push_back(n);
      }
      p.parseComma();
      isgeneral = 0;
    }
    auto elemID = p.parseIdentifier();
    auto fc = typeTable::get().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseGreaterThan();
    if(isgeneral){
      return ctx->getType<tensor_t>(elemType);
    }
    return ctx->getType<tensor_t>(elemType, shapes);
  }
};

void registerTypes(){
  REGISTER_TYPE(integer, "integer");
  REGISTER_TYPE(natureNumber, "natureNumber");
  REGISTER_TYPE(realNumber, "realNumber");
  REGISTER_TYPE(rationalNumber, "rationalNumber");
  REGISTER_TYPE(irrationalNumber, "irrationalNumber");
  REGISTER_TYPE(tensor_t, "tensor");
}

}
#endif
