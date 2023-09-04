#ifndef LGF_MODULE_LINEARALG_TYPES_H
#define LGF_MODULE_LINEARALG_TYPES_H
#include "libs/builtin/types.h"
#include "lgf/printer.h"

namespace lgf{
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
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseGreaterThan();
    if(isgeneral){
      return ctx->getType<tensor_t>(elemType);
    }
    return ctx->getType<tensor_t>(elemType, shapes);
  }
};
}
#endif