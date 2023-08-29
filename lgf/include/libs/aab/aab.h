
#ifndef LGF_MATHLIB_AAB_H
#define LGF_MATHLIB_AAB_H
#include "ops.h"
#include "lgf/LGFModule.h"

namespace lgf::math{
class AABModule : public LGFModule {
    public:
    AABModule() = default;
    virtual void registerTypes(){}

};
}

#endif