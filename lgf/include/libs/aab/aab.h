
#ifndef LGF_MATHLIB_AAB_H
#define LGF_MATHLIB_AAB_H
#include "ops.h"
#include "lgf/LGFModule.h"
#include "types.h"
#include "passes.h"

namespace lgf{
class AABModule : public LGFModule {
    public:
    AABModule() = default;
    static void registerTypes(LGFContext *ctx){
        ctx->registerType<integer>("integer");
        ctx->registerType<natureNumber>("natureNumber");
        ctx->registerType<realNumber>("realNumber");
        ctx->registerType<rationalNumber>("rationalNumber");
        ctx->registerType<irrationalNumber>("irrationalNumber");
        ctx->registerType<infinitesimal>("infinitesimal");
    }
};
}

#endif