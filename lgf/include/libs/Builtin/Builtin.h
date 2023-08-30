
#ifndef LGF_MAIN_HEADER
#define LGF_MAIN_HEADER

#include "types.h"
#include "ops.h"
#include "lgf/LGFContext.h"
#include "lgf/LGFModule.h"
#include "lgf/pass.h"

namespace lgf{

class LGFBaseModule : public LGFModule {
    public:
    LGFBaseModule() = default;
    static void registerTypes(LGFContext* ctx) {
        ctx->getTypeTable().registerType<variable>("variable");
        ctx->getTypeTable().registerType<intType>("int");
        ctx->getTypeTable().registerType<doubleType>("double");
        ctx->getTypeTable().registerType<listType>("listType");
    }
    virtual void pipeline(passManager& pm){}
};

}

#endif