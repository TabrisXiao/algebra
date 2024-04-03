
#ifndef LGF_INTERFACE_VAR_H
#define LGF_INTERFACE_VAR_H
#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/aab/aab.h"

namespace lgi{

class var {
    public:
    var(bool init = 1) {
        if(!init) return;
        auto& ctx = canvas::get().getContext();
        v = canvas::get().getPainter().paint<lgf::declOp>(ctx.getType<lgf::variable>())->output();
    }
    var(lgf::value* val){
        v = val;
    }
    void operator = (const var& other){
        v = other.v;
    }
    var operator+(const var& other){
        auto& ctx = canvas::get().getContext();
        auto res = canvas::get().getPainter().paint<lgf::AAB::sumOp>(v, other.v)->output();
        return var(res);
    } 
    var operator*(const var& other){
        auto& ctx = canvas::get().getContext();
        auto res = canvas::get().getPainter().paint<lgf::AAB::productOp>(v, other.v)->output();
        return var(res);
    }
    lgf::value* value() const{
        return v;
    }
    protected:
    lgf::value *v = nullptr;
};

} // namespace lgi

#endif