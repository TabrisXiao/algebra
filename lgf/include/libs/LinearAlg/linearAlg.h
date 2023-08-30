#ifndef LGF_MATHLIB_LINEARALG_H
#define LGF_MATHLIB_LINEARALG_H
#include "ops.h"
#include "lgf/LGFModule.h"
#include "libs/Builtin/types.h"
#include "types.h"

namespace lgf{
class LinearAlgModule : public LGFModule {
    public:
    LinearAlgModule() = default;
    static void registerTypes(LGFContext *ctx){
        ctx->registerType<tensor_t>("tensor");
    }
};
}

#endif