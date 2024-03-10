
#ifndef LGF_BUILTIN_TYPES_H_
#define LGF_BUILTIN_TYPES_H_
#include "lgf/type.h"
#include "lgf/LGFContext.h"

namespace lgf{

class variable : public lgf::type_t {
    public:
    using desc_t = descriptor;
    static inline const sid_t sid= "var";
    variable(){};
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<variable>();
    }
};

class reference_t : public lgf::type_t {
  public:
  using desc_t = lgf::descriptor;
  static inline const sid_t sid= "reference";
  reference_t() = default;
  static type_t parse(liteParser& paser, LGFContext* ctx){
    return ctx->getType<reference_t>();
  }
};

class intType: public variable {
    public:
    using desc_t = lgf::descriptor;
    static inline const sid_t sid= "int";
    intType() = default;
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<intType>();
    }
};

class doubleType: public intType {
    public:
    using desc_t = lgf::descriptor;
    static inline const sid_t sid= "double";
    doubleType() = default;
    static type_t parse(liteParser& paser, LGFContext* ctx){
      return ctx->getType<doubleType>();
    }
};

}

#endif