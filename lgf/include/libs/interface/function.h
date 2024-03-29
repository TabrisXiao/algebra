
#ifndef LGF_INTERFACE_FUNCTION_H
#define LGF_INTERFACE_FUNCTION_H

#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/fa/types.h"
#include "libs/fa/ops.h"
#include "libs/aab/aab.h"

namespace lgi::function{

class variable {
    public:
    variable(){
        auto& ctx = canvas::get().getContext();
        v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::variable>())->output();
    }
    void operator = (const variable& other){
        v = other.v;
    }
    lgf::value* operator+(const variable& other){
        auto& ctx = canvas::get().getContext();
        auto res = canvas::get().getPainter().paint<lgf::AAB::sumOp>(v, other.v)->output();
        return res;
    } 
    lgf::value* operator*(const variable& other){
        auto& ctx = canvas::get().getContext();
        auto res = canvas::get().getPainter().paint<lgf::AAB::productOp>(v, other.v)->output();
        return res;
    }
    lgf::value *v = nullptr;
};

class set {
    public:
    set(){
        auto& ctx = canvas::get().getContext();
        v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::set_t>())->output();
    }
    void operator = (const variable& other){
        v = other.v;
    }
    lgf::value *v = nullptr;
};


} // namespace  lgi::function

#endif