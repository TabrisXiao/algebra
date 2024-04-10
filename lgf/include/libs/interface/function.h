
#ifndef LGF_INTERFACE_FUNCTION_H
#define LGF_INTERFACE_FUNCTION_H

#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/functional/functional.h"
#include "variable.h"

namespace lgi::function{
using namespace lgi;
variable cos(const variable& x){
    auto& ctx = canvas::get().get_context();
    auto res = canvas::get().get_painter().paint<lgf::funcCosOp>(x.node());
    return variable(res);
}

variable sin(const variable& x){
    auto& ctx = canvas::get().get_context();
    auto res = canvas::get().get_painter().paint<lgf::funcSineOp>(x.node());
    return variable(res);
}

variable power(const variable& x, double n){
    auto& ctx = canvas::get().get_context();
    auto res = canvas::get().get_painter().paint<lgf::funcPowerOp>(x.node(), n);
    return variable(res);
}

variable exp(const variable& x){
    auto& ctx = canvas::get().get_context();
    auto res = canvas::get().get_painter().paint<lgf::funcExpOp>(x.node());
    return variable(res);
}

class set {
    public:
    set(bool is_empty = false){
        auto& ctx = canvas::get().get_context();
        if(is_empty){
            v = canvas::get().get_painter().paint<lgf::declOp>(ctx.get_desc<lgf::empty_set_t>());
        }else{
            v = canvas::get().get_painter().paint<lgf::declOp>(ctx.get_desc<lgf::set_desc>());
        }
    }
    void operator = (const set& other){
        v = other.v;
    } 
    lgf::node *v = nullptr;
};


} // namespace  lgi::function

#endif