
#ifndef LGF_INTERFACE_FUNCTION_H
#define LGF_INTERFACE_FUNCTION_H

#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/fa/types.h"
#include "libs/fa/ops.h"
#include "libs/aab/aab.h"

namespace lgi::function{

class realNumber {
    public:
    realNumber(){
        auto& ctx = canvas::get().getContext();
        v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::realNumber>())->output();
    }
    void operator = (const realNumber& other){
        v = other.v;
    }
    lgf::value* operator+(const realNumber& other){
        auto& ctx = canvas::get().getContext();
        auto res = canvas::get().getPainter().paint<lgf::AAB::sumOp>(v, other.v)->output();
        return res;
    } 
    lgf::value* operator*(const realNumber& other){
        auto& ctx = canvas::get().getContext();
        auto res = canvas::get().getPainter().paint<lgf::AAB::productOp>(v, other.v)->output();
        return res;
    }
    lgf::value *v = nullptr;
};

} // namespace  lgi::function

#endif