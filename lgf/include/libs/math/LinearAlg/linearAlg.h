#ifndef LGF_MATHLIB_LINEARALG_H
#define LGF_MATHLIB_LINEARALG_H
#include "ops.h"
#include "lgf/LGFModule.h"

namespace lgf::math{
class LinearAlgModule : public LGFModule {
    public:
    LinearAlgModule() = default;
    virtual void registerTypes(){}
};
}

#endif