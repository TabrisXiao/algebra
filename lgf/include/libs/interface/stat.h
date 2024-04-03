
#ifndef LGF_INTERFACE_STAT_H
#define LGF_INTERFACE_STAT_H

#include "function.h"
#include "libs/stat/types.h"
#include "variable.h"

namespace lgi::stat{
using namespace lgi;
class randomVariable : public var{
    public:
    randomVariable(bool init = 1): var(false) {
        if(!init) return;
        auto& ctx = canvas::get().getContext();
        v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::randomVariable>())->output();
    }
};

class normalVariable : public randomVariable{
    public:
    normalVariable(double mean = 0, double variance = 1):
    randomVariable(false) {
        auto& ctx = canvas::get().getContext();
        v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::normalVariable>(mean, variance))->output();
    }
};

} // namespace lgi::stat

#endif