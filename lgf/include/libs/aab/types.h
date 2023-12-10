#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "libs/Builtin/types.h"
#include "lgf/typeTable.h"

namespace lgf{

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

class infinitesimalImpl : public lgf::derivedTypeImpl { 
  public: 
  infinitesimalImpl(lgf::type_t elemType_)
  : derivedTypeImpl("infinitesimal", elemType_) {}
};

// class infinitesimalImpl : public lgf::typeImpl { 
//   public: 
//   infinitesimalImpl(lgf::type_t elemType_)
//   : typeImpl("infinitesimal")
//   , elemType(elemType_) {}

//   virtual std::string represent(){
//     printer p;
//     p<<id<<"<";
//     p<<elemType.represent()<<">";
//     return p.dump();
//   }
//   lgf::type_t elemType;
// };

class infinitesimal : public lgf::variable {
  public:
  infinitesimal() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType){
    return std::move(std::make_unique<infinitesimalImpl>(elemType));
  }
  type_t getElemType(){ return dynamic_cast<infinitesimalImpl*>(impl)->getBaseType(); }

  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    p.parseLessThan();
    auto elemID = p.parseIdentifier();
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseGreaterThan();
    return ctx->getType<infinitesimal>(elemType);
  }
};

class unitImpl : public lgf::derivedTypeImpl { 
  public: 
  unitImpl(lgf::type_t elemType_)
  : derivedTypeImpl("unit", elemType_) {}
}; 

class zeroImpl : public lgf::derivedTypeImpl { 
  public: 
  zeroImpl(lgf::type_t elemType_)
  : derivedTypeImpl("zero", elemType_) {}
};

class unit_t : public lgf::variable {
  public:
  unit_t() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType){
    return std::move(std::make_unique<unitImpl>(elemType));
  }
  type_t getElemType(){ return dynamic_cast<unitImpl*>(impl)->getBaseType(); }

  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    p.parseLessThan();
    auto elemID = p.parseIdentifier();
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseGreaterThan();
    return ctx->getType<unit_t>(elemType);
  }
};

class zero_t : public lgf::variable {
  public:
  zero_t() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType){
    return std::move(std::make_unique<zeroImpl>(elemType));
  }
  type_t getElemType(){ return dynamic_cast<zeroImpl*>(impl)->getBaseType(); }

  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    p.parseLessThan();
    auto elemID = p.parseIdentifier();
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseGreaterThan();
    return ctx->getType<zero_t>(elemType);
  }
};

}
#endif
