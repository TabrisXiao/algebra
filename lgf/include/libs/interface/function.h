
#ifndef LGF_INTERFACE_FUNCTION_H
#define LGF_INTERFACE_FUNCTION_H

#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/fa/types.h"
#include "libs/fa/ops.h"
#include "libs/aab/aab.h"
#include "variable.h"

namespace lgi::function{
using namespace lgi;
var cos(const var& x){
    auto& ctx = canvas::get().getContext();
    auto res = canvas::get().getPainter().paint<lgf::funcCosOp>(x.value())->output();
    return var(res);
}

var sin(const var& x){
    auto& ctx = canvas::get().getContext();
    auto res = canvas::get().getPainter().paint<lgf::funcSineOp>(x.value())->output();
    return var(res);
}

var power(const var& x, double n){
    auto& ctx = canvas::get().getContext();
    auto res = canvas::get().getPainter().paint<lgf::powerOp>(x.value(), n)->output();
    return var(res);
}

var exp(const var& x){
    auto& ctx = canvas::get().getContext();
    auto res = canvas::get().getPainter().paint<lgf::expOp>(x.value())->output();
    return var(res);
}

class set {
    public:
    set(bool is_empty = false){
        auto& ctx = canvas::get().getContext();
        if(is_empty){
            v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::empty_set_t>())->output();
        }else{
            v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::set_t>())->output();
        }
    }
    void operator = (const var& other){
        v = other.value();
    } 
    lgf::value *v = nullptr;
};


} // namespace  lgi::function

#endif