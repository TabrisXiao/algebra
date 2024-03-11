
#ifndef LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#define LGF_FUNCTIONAL_ANALYSIS_TYPE_H
#include "libs/aab/types.h"

namespace lgf{

class realNumber: public lgf::variable {
    public:
    realNumber() = default;
    static type_t parse(lgf::liteParser& paser, lgf::LGFContext* ctx){
      return ctx->getType<realNumber>();
    }
};

} // namespace lgf

#endif