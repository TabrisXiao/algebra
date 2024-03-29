
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#include "libs/aab/types.h"

namespace lgf{

class real_t: public lgf::variable {
    public:
    static inline const sid_t sid= "real";
    real_t() = default;
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<real_t>();
    }
};

class set_desc : public descriptor {
    public:
    set_desc(){}
    virtual std::string representType()const override{
        return getSID();
    }
};

class set_t : public lgf::variable {
    public:
    using desc_t = set_desc;
    static inline const sid_t sid= "set";
    set_t() = default;
};

class empty_set_t: public lgf::variable{
    public:
    using desc_t = descriptor;
    static inline const sid_t sid= "empty-set";
    empty_set_t() = default;
};

class sigma_algebra_t: public lgf::variable{ 
    public:
    using desc_t = descriptor;
    static inline const sid_t sid= "sigma-algebra";
    sigma_algebra_t() = default;
};

} // namespace lgf

#endif