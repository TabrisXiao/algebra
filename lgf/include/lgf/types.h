
#ifndef INTERNAL_TYPES_H_
#define INTERNAL_TYPES_H_
#include "type.h"

namespace lgf{

class variable : public lgf::type_t {
    public:
    variable() = default;
    static std::unique_ptr<typeImpl> createImpl(){
      return std::move(std::make_unique<typeImpl>("Variable"));
    }
};

class intType: public variable {
    public:
    intType() = default;
    static std::unique_ptr<typeImpl> createImpl(){
      return std::move(std::make_unique<typeImpl>("int"));
    }
};

class doubleType: public variable {
    public:
    doubleType() = default;
    static std::unique_ptr<typeImpl> createImpl(){
      return std::move(std::make_unique<typeImpl>("double"));
    }
};

class listTypeImpl : public typeImpl {
  public: 
  listTypeImpl(type_t elemType_, int dim) 
  : typeImpl("list")
  , elemType(elemType_)
  , size(dim) {}
  virtual std::string represent(){
    printer p;
    p<<id<<"<"<<elemType.represent()<<", "<<std::to_string(size)<<">";
    return p.dump();
  }
  int size=0;
  type_t elemType;
};

class listType : public type_t {
  public:
  listType() = default;
  static std::unique_ptr<typeImpl> createImpl(type_t elemType, int size){
    return std::move(std::make_unique<listTypeImpl>(elemType, size));
  }
  int size(){
    return dynamic_cast<listTypeImpl*>(impl)->size;
  }
};


}

#endif