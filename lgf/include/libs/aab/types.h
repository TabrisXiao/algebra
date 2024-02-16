#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "libs/Builtin/types.h"
#include "lgf/typeTable.h"

namespace lgf{

using dim_t = uint32_t;

class algebraAxiom: public typeMarker<uint32_t> {
    public:
    enum trait: uint8_t{
      add_commutable = 0,
      multiply_commutable = 1,
      //associative and distributive are for multiply and add operation
      associative = 2,
      distributive = 3,
    };
    algebraAxiom(): typeMarker<uint32_t>(32){}
    void initAsField(){
      mark(add_commutable);
      mark(multiply_commutable);
      mark(associative);
      mark(distributive);
    }
    void initAsRing(){
      mark(add_commutable);
      mark(associative);
      mark(distributive);
    }
};

class algebraVariableImpl : public descriptor, public algebraAxiom {
    public:
    algebraVariableImpl(std::string sid): descriptor(sid){
    }
};

class fieldVariableImpl : public algebraVariableImpl {
    public:
    fieldVariableImpl(std::string sid): algebraVariableImpl(sid){
      initAsField();
    }
};

class realNumber: public lgf::variable {
    public:
    realNumber() = default;
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<realNumber>();
    }
};
class vectorDesc : public lgf::descriptor, public algebraAxiom {
  public:
  vectorDesc(type_t elemType_, uint32_t dim_)
  : lgf::descriptor("vector")
  , elemType(elemType_)
  , dim(dim_){
    initAsRing();
  }
  virtual std::string represent(){
    return id+"<"+elemType.represent()+","+std::to_string(dim)+">";
  }
  lgf::type_t elemType;
  dim_t dim;
};

class tensorDesc : public lgf::descriptor, public algebraAxiom{
  public:
  tensorDesc(type_t elem_t, std::vector<dim_t> rank_)
  : lgf::descriptor("tensor")
  , elemType(elem_t) {
    dims.swap(rank_);
    initAsRing();
  }
  virtual std::string represent(){
    return id+"<"+elemType.represent()+","+dimRepresent()+">";
  }
  std::string dimRepresent(){
    std::string ret;
    for(auto& d : dims){
      ret += std::to_string(d)+"x";
    }
    ret.pop_back();
    return ret;
  }
  std::vector<dim_t> dims;
  lgf::type_t elemType;
};


class vectorType : public lgf::variable {
  public:
  using desc_t = vectorDesc;
  vectorType() = default;
  type_t getElemType(){ return dynamic_cast<vectorDesc*>(desc)->elemType; }
  dim_t getDim(){ return dynamic_cast<vectorDesc*>(desc)->dim; }

  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    p.parseLessThan();
    auto elemID = p.parseIdentifier();
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseComma();
    auto dim = p.parseInteger();
    p.parseGreaterThan();
    return ctx->getType<vectorType>(elemType, uint32_t(dim));
  }
};

// class tensorType : public lgf::variable {
//   public:
//   tensorType() = default;
//   static std::unique_ptr<lgf::typeImpl> createImpl(type_t vecType, uint32_t rank){
//     return std::move(std::make_unique<tensorImpl>(vecType, rank));
//   }
//   type_t getElementType(){ return dynamic_cast<tensorImpl*>(impl)->elemType; }
//   uint32_t getRank(){ return dynamic_cast<tensorImpl*>(impl)->dims.size(); }

//   static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
//     p.parseLessThan();
//     auto vecID = p.parseIdentifier();
//     auto fc = ctx->getTypeTable().findParser(vecID);
//     auto vecType = fc(p, ctx);
//     p.parseComma();
//     auto rank = p.parseInteger();
//     p.parseGreaterThan();
//     return ctx->getType<tensorType>(vecType, dim_t(rank));
//   }
// };

// class integer: public lgf::variable {
//     public:
//     integer() {};
//     static std::unique_ptr<lgf::typeImpl> createImpl(){
//       return std::move(std::make_unique<lgf::fieldVariableImpl>("integer"));
//     }
//     static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
//       return ctx->getType<integer>();
//     }
// };

// class natureNumber: public lgf::variable {
//     public:
//     natureNumber() = default;
//     static std::unique_ptr<lgf::typeImpl> createImpl(){
//       return std::move(std::make_unique<lgf::fieldVariableImpl>("natureNumber"));
//     }
//     static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
//       return ctx->getType<natureNumber>();
//     }
// };

// class rationalNumber: public lgf::variable {
//     public:
//     rationalNumber() = default;
//     static std::unique_ptr<lgf::typeImpl> createImpl(){
//       return std::move(std::make_unique<lgf::fieldVariableImpl>("rationalNumber"));
//     }
//     static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
//       return ctx->getType<rationalNumber>();
//     }
// };

// class irrationalNumber: public lgf::variable {
//   public:
//   irrationalNumber() = default;
//   static std::unique_ptr<lgf::typeImpl> createImpl(){
//     return std::move(std::make_unique<lgf::fieldVariableImpl>("irrationalNumber"));
//   }
//   static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
//     return ctx->getType<irrationalNumber>();
//   }
// };

// class infinitesimalImpl : public lgf::derivedTypeImpl, public fieldVariableImpl { 
//   public: 
//   infinitesimalImpl(lgf::type_t elemType_)
//   : derivedTypeImpl("infinitesimal", elemType_), 
//   fieldVariableImpl("infinitesimal") {}
// };

class unitDesc : public lgf::derivedTypeDesc, public algebraAxiom { 
  public: 
  unitDesc(lgf::type_t elemType_)
  : derivedTypeDesc("unit", elemType_) {
    initAsField();
  }
}; 

class zeroDesc : public lgf::derivedTypeDesc, public algebraAxiom { 
  public: 
  zeroDesc(lgf::type_t elemType_)
  : derivedTypeDesc("zero", elemType_) {
    initAsField();
  }
};

class unit_t : public lgf::variable {
  public:
  using desc_t = unitDesc;
  unit_t() = default;
  type_t getElemType(){ return dynamic_cast<unitDesc*>(desc)->getBaseType(); }

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
  using desc_t = zeroDesc;
  zero_t() = default;

  type_t getElemType(){ return dynamic_cast<zeroDesc*>(desc)->getBaseType(); }

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
