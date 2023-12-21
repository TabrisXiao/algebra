#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "libs/Builtin/types.h"
#include "lgf/typeTable.h"

namespace lgf{

// class algebraFeature: public typeMarker<uint32_t> {
//     public:
//     enum trait: uint8_t{
//       add_commutable = 0,
//       multiply_commutable = 1,
//       associative and distributive are for multiply and add operation
//       associative = 2,
//       distributive = 3,
//     };
//     algebraFeature(): typeMarker<uint32_t>(32){}
//     void initAsField(){
//       mark(commutable);
//       mark(associative);
//       mark(distributive);
//     }
//     void initAsRing(){}
// };

// class algebraVariableImpl : public typeImpl, public algebraFeature {
//     public:
//     algebraVariableImpl(std::string sid): typeImpl(sid){}
//     void 
// };

class fieldVariableImpl : public lgf::typeImpl {
    public:
    fieldVariableImpl(std::string sid): typeImpl(sid){
    }
};

class integer: public lgf::variable {
    public:
    integer() {};
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::fieldVariableImpl>("integer"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<integer>();
    }
};

class natureNumber: public lgf::variable {
    public:
    natureNumber() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::fieldVariableImpl>("natureNumber"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<natureNumber>();
    }
};

class realNumber: public lgf::variable {
    public:
    realNumber() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::fieldVariableImpl>("realNumber"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<realNumber>();
    }
};

class rationalNumber: public lgf::variable {
    public:
    rationalNumber() = default;
    static std::unique_ptr<lgf::typeImpl> createImpl(){
      return std::move(std::make_unique<lgf::fieldVariableImpl>("rationalNumber"));
    }
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<rationalNumber>();
    }
};

class irrationalNumber: public lgf::variable {
  public:
  irrationalNumber() = default;
  static std::unique_ptr<lgf::typeImpl> createImpl(){
    return std::move(std::make_unique<lgf::fieldVariableImpl>("irrationalNumber"));
  }
  static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
    return ctx->getType<irrationalNumber>();
  }
};

class infinitesimalImpl : public lgf::derivedTypeImpl, public fieldVariableImpl { 
  public: 
  infinitesimalImpl(lgf::type_t elemType_)
  : derivedTypeImpl("infinitesimal", elemType_), 
  fieldVariableImpl("infinitesimal") {}
};

// class infinitesimal : public lgf::variable {
//   public:
//   infinitesimal() = default;
//   static std::unique_ptr<lgf::typeImpl> createImpl(type_t elemType){
//     return std::move(std::make_unique<infinitesimalImpl>(elemType));
//   }
//   type_t getElemType(){ return dynamic_cast<infinitesimalImpl*>(impl)->getBaseType(); }

//   static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
//     p.parseLessThan();
//     auto elemID = p.parseIdentifier();
//     auto fc = ctx->getTypeTable().findParser(elemID);
//     auto elemType = fc(p, ctx);
//     p.parseGreaterThan();
//     return ctx->getType<infinitesimal>(elemType);
//   }
// };

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
