
#ifndef INTERNAL_TYPES_H_
#define INTERNAL_TYPES_H_
#include "type.h"
#include "LGFContext.h"
#include "typeTable.h"

namespace lgf{

class variable : public lgf::type_t {
    public:
    variable() = default;
    static std::unique_ptr<typeImpl> createImpl(){
      return std::move(std::make_unique<typeImpl>("Variable"));
    }
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<variable>();
    }
};

class reference_t : public lgf::type_t {
  public:
  reference_t() = default;
  static std::unique_ptr<typeImpl> createImpl(){
    return std::move(std::make_unique<typeImpl>("ref"));
  }
  static type_t parse(liteParser& paser, LGFContext* ctx){
    return ctx->getType<variable>();
  }
};

class intType: public variable {
    public:
    intType() = default;
    static std::unique_ptr<typeImpl> createImpl(){
      return std::move(std::make_unique<typeImpl>("int"));
    }
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<intType>();
    }
};


class doubleType: public intType {
    public:
    doubleType() = default;
    static std::unique_ptr<typeImpl> createImpl(){
      return std::move(std::make_unique<typeImpl>("double"));
    }
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<doubleType>();
    }
};

class listTypeImpl : public typeImpl {
  public: 
  listTypeImpl(type_t elemType_, int dim) 
  : typeImpl("list")
  , elemType(elemType_)
  , size(dim) {}
  virtual std::string represent(){
    return id+"<"+elemType.represent()+", "+std::to_string(size)+">";
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
  static type_t parse(liteParser& paser, LGFContext* ctx){
    paser.parseLessThan();
    int size = paser.parseNumber();
    paser.parseComma();
    auto elemID = paser.parseIdentifier();
    auto fc = typeTable::get().findParser(elemID);
    auto elemType = fc(paser, ctx);
    paser.parseGreaterThan();
    return ctx->getType<listType>(elemType, size);
  }
};

}

#endif