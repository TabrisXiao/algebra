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

class algebraVariableImpl : public algebraAxiom {
    public:
    algebraVariableImpl(std::string sid){
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

class sequenceDesc : public lgf::descriptor {
  public:
  sequenceDesc() = default;
  sequenceDesc(std::vector<lgf::type_t>& types_)
  : types(types_){}
  template<typename ...ARGS>
  sequenceDesc(ARGS ...args)
  : types({args...}){}
  virtual std::string representType()const override{
    std::string str= getSID()+", size="+std::to_string(types.size())+" {";
    for(auto & t: types){
      str += t.represent()+",";
    }
    str.pop_back();
    str+="}";
    return str;
  }
  const std::vector<lgf::type_t>& getTypes() const { return types; } 
  private:
  std::vector<lgf::type_t> types;
  dim_t dim;
};

class sequenceType: public lgf::variable {
  public:
  using desc_t = sequenceDesc;
  static inline const std::string sid= "sequence";
  sequenceType() = default;
  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    THROW("sequenceType::parse not implemented")
    return ctx->getType<sequenceType>();
  }
};

class vectorDesc : public lgf::descriptor, public algebraAxiom {
  public:
  vectorDesc(type_t elemType_, dim_t dim_)
  : elemType(elemType_)
  , dim(dim_){
    initAsRing();
  }
  virtual std::string representType() const override {
    return getSID()+"<"+elemType.represent()+","+std::to_string(dim)+">";
  }
  lgf::type_t elemType;
  dim_t dim;
};


class vectorType : public lgf::variable {
  public:
  using desc_t = vectorDesc;
  static inline const std::string sid= "vector";
  vectorType() {};
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

class tensorDesc : public lgf::descriptor, public algebraAxiom{
  public:
  tensorDesc(type_t elem_t, std::vector<dim_t> rank_){
    elemType = elem_t;
    dims = rank_;
    initAsRing();
  }
  virtual std::string representType() const override {
    return getSID()+"<"+elemType.representType()+","+dimRepresent()+">";
  }
  std::string dimRepresent() const {
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

class tensorType : public lgf::variable {
  public:
  using desc_t = tensorDesc;
  static inline const sid_t sid = "tensor";
  tensorType() = default;
  type_t getElemType(){ return dynamic_cast<tensorDesc*>(desc)->elemType; }
  std::vector<dim_t>& getDims() const { return dynamic_cast<tensorDesc*>(desc)->dims; }

  static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
    p.parseLessThan();
    auto elemID = p.parseIdentifier();
    auto fc = ctx->getTypeTable().findParser(elemID);
    auto elemType = fc(p, ctx);
    p.parseComma();
    std::vector<dim_t> dims;
    p.parseLessThan();
    while(p.getCurToken() != liteParser::token('>')){
      dims.push_back(p.parseNumber<uint32_t>());
      if(p.getCurToken() == liteParser::token(',')){
        p.parseComma();
      }
    }
    p.parseGreaterThan();
    return ctx->getType<tensorType>(elemType, dims);
  }
};

// class unitDesc : public lgf::derivedTypeDesc, public algebraAxiom { 
//   public: 
//   unitDesc(const sid_t sid, lgf::type_t elemType_)
//   : derivedTypeDesc(sid, elemType_) {
//     initAsField();
//   }
// }; 

// class zeroDesc : public lgf::derivedTypeDesc, public algebraAxiom { 
//   public: 
//   zeroDesc(const sid_t sid, lgf::type_t elemType_)
//   : derivedTypeDesc(sid, elemType_) {
//     initAsField();
//   }
// };

// class unit_t : public lgf::variable {
//   public:
//   using desc_t = unitDesc;
//   unit_t() = default;
//   type_t getElemType(){ return dynamic_cast<unitDesc*>(desc)->getBaseType(); }

//   static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
//     p.parseLessThan();
//     auto elemID = p.parseIdentifier();
//     auto fc = ctx->getTypeTable().findParser(elemID);
//     auto elemType = fc(p, ctx);
//     p.parseGreaterThan();
//     return ctx->getType<unit_t>(elemType);
//   }
// };

// class zero_t : public lgf::variable {
//   public:
//   using desc_t = zeroDesc;
//   zero_t() = default;

//   type_t getElemType(){ return dynamic_cast<zeroDesc*>(desc)->getBaseType(); }

//   static type_t parse(lgf::liteParser& p, lgf::LGFContext* ctx){
//     p.parseLessThan();
//     auto elemID = p.parseIdentifier();
//     auto fc = ctx->getTypeTable().findParser(elemID);
//     auto elemType = fc(p, ctx);
//     p.parseGreaterThan();
//     return ctx->getType<zero_t>(elemType);
//   }
// };

}
#endif
