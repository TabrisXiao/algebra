
#ifndef LGF_STAT_OPS_H
#define LGF_STAT_OPS_H
#include "libs/aab/ops.h"

namespace  lgf::stat
{

class expectationOp : public AAB::mappingOp{
    public:
    expectationOp() :  mappingOp("stat::E"){}
    static expectationOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new expectationOp();
        op->addArgument(x);
        op->createValue(x->getType(), "");
        return op;
    }
    virtual std::string represent() const override{
        return "E("+inputValue(0)->represent()+")";
    }
};

} // namespace  lgf::stat


#endif