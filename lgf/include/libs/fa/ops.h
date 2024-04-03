
#ifndef LGF_FUNCTIONAL_ANALYSIS_OPS_H
#define LGF_FUNCTIONAL_ANALYSIS_OPS_H
#include "libs/aab/ops.h"
#include "libs/fa/types.h"

namespace lgf{

class funcSineOp : public AAB::mappingOp{
    public:
    funcSineOp() :  mappingOp("functional::sine"){}
    static funcSineOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new funcSineOp();
        op->addArgument(x);
        op->createValue(x->getType(), "");
        return op;
    }
};

class funcCosOp : public AAB::mappingOp{
    public:
    funcCosOp(): mappingOp("functional::cos"){}
    static funcCosOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new funcCosOp();
        op->addArgument(x);
        op->createValue(x->getType(), "");
        return op;
    }
};

class powerOp : public AAB::mappingOp
{
    public:
    powerOp() : mappingOp("functional::power"){}
    static powerOp* build(lgf::LGFContext* ctx, lgf::value* x, double n){
        auto op = new powerOp();
        op->addArgument(x);
        op->setPower(n);
        op->createValue(x->getType(), "");
        return op;
    }
    void setPower(double n){ p = n; }
    double power(){ return p; }
    lgf::value* x(){ return inputValue(0); }
    double p=1;
};

class expOp : public AAB::mappingOp {
    public:
    expOp() : mappingOp("functional::exp"){}
    static expOp* build(LGFContext* ctx, value* power){
        auto op = new expOp();
        op->addArgument(power);
        op->createValue(power->getType(), "");
        return op;
    }
    value* power(){
        return inputValue(0);
    }
    void inferType(LGFContext* ctx) override{
        // using the input type as output type
        output()->setType(power()->getType()); 
    }
};

class partialDifferentiateOp : public AAB::mappingOp {
    public:
    partialDifferentiateOp() : mappingOp("functional::PartialDifferentiate"){}
    static partialDifferentiateOp* build(LGFContext* ctx, value* func, value* var){
        auto op = new partialDifferentiateOp();
        op->addArgument(func, var);
        op->createValue(func->getType(), "");
        return op;
    }
    value* func(){ return inputValue(0); }
    value* var(){ return inputValue(1); }
};

class differentiateOp : public AAB::mappingOp {
    public:
    differentiateOp() : mappingOp("functional::differentiate"){}
    static differentiateOp* build(LGFContext* ctx, value* input, value* target){
        auto op = new differentiateOp();
        op->addArgument(input, target);
        op->createValue(input->getType(), "");
        return op;
    }
    value* input(){ return inputValue(0); }
    value* target(){ return inputValue(1); }
};

class unionOp: public operation{
    public:
    unionOp() : operation("functional::union"){}
    template<typename ...ARGS>
    static unionOp* build(LGFContext* ctx, ARGS ...args){
        auto op = new unionOp();
        op->registerInput(args...);
        op->createValue(op->inputValue()->getType());
        return op;
    }
};

class intersectOp : public operation{
    public:
    intersectOp() : operation("functional::intersect"){}
    template<typename ...ARGS>
    static intersectOp* build(LGFContext* ctx, ARGS ...args){
        auto op = new intersectOp();
        op->registerInput(args...);
        op->createValue(op->inputValue()->getType());
        return op;
    }
};

} // namespace lgf

#endif