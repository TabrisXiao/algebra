
#ifndef LGF_BUILTIN_TYPES_H_
#define LGF_BUILTIN_TYPES_H_
#include "lgf/type.h"
#include "lgf/LGFContext.h"

namespace lgf{

class variable : public lgf::type_t {
    public:
    using desc_t = lgf::descriptor;
    variable() = default;
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<variable>();
    }
};

class derivedTypeDesc : public lgf::descriptor {
  public:
  derivedTypeDesc(std::string id_, type_t baseType_)
  : lgf::descriptor(id_)
  , baseType(baseType_){}

  virtual std::string represent(){
    return id+"<"+baseType.represent()+">";
  }

  lgf::type_t getBaseType(){
    return baseType;
  }
  type_t baseType;
};

class reference_t : public lgf::type_t {
  public:
  using desc_t = lgf::descriptor;
  reference_t() = default;
  static type_t parse(liteParser& paser, LGFContext* ctx){
    return ctx->getType<reference_t>();
  }
};

class intType: public variable {
    public:
    intType() = default;
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<intType>();
    }
};


class doubleType: public intType {
    public:
    doubleType() = default;
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<doubleType>();
    }
};

class listDesc : public descriptor {
  public: 
  listDesc(type_t elemType_, int dim) 
  : descriptor("list")
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
  using desc_t = listDesc;
  listType() = default;
  int size(){
    return dynamic_cast<listDesc*>(desc)->size;
  }
  static type_t parse(liteParser& paser, LGFContext* ctx){
    paser.parseLessThan();
    int size = int(paser.parseNumber());
    paser.parseComma();
    auto elemID = paser.parseIdentifier();
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(paser, ctx);
    paser.parseGreaterThan();
    return ctx->getType<listType>(elemType, size);
  }
};

}

#endif