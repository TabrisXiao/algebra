
#ifndef MATH_AAB_OPS_H
#define MATH_AAB_OPS_H
#include "lgf/operation.h"
#include "libs/builtin/types.h"
#include "lgf/group.h"

namespace lgf::AAB{

// ---------- addOp ----------
class addOp : public lgf::operation
{
    public:
    addOp() : operation("AAB::add") {}
    static addOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new addOp();
        op->registerInput(lhs, rhs);
        op->createValue(ctx->getType<lgf::variable>(), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }
    lgf::value* output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0)->represent()<<" + "<<inputValue(1)->represent();
        return p.dump();
    }
};

// ---------- minusOp ----------
class sumOp : public lgf::operation {
    public:
    sumOp() : operation("AAB::sumOp") {}
    static sumOp* build(lgf::LGFContext* ctx, std::vector<value*>& vec){
        auto op = new sumOp();
        op->registerInputs(vec);
        op->createValue(vec[0]->getType(), "");
        return op;
    }
    lgf::value* input(int i=0){ return inputValue(i); }
    lgf::value* output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<representInputs();
        return p.dump();
    }
};

// ---------- negativeOp ----------
class negativeOp : public lgf::operation
{
    public:
    public:
    negativeOp() : operation("AAB::negative") {}
    static negativeOp* build(lgf::LGFContext* ctx, lgf::value* input){
        auto op = new negativeOp();
        op->registerInput(input);
        op->createValue(input->getType(), "");
        return op;
    }
    lgf::value* input(){ return inputValue(0); }
    lgf::value* output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<input()->represent();
        return p.dump();
    }
};

// ---------- minusOp ----------
class minusOp : public lgf::operation, public normalizer
{
    public:
    minusOp() : operation("AAB::minus") {}
    static minusOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new minusOp();
        op->registerInput(lhs, rhs);
        op->createValue(ctx->getType<lgf::variable>(), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }
    lgf::value* output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<lhs()->represent()<<" - "<<rhs()->represent();
        return p.dump();
    }
    virtual bool rewrite(painter& p, operation* op){
        std::cout<<"---------------find minus renomalizer!"<<std::endl;
        return 0;
    }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
    public:
    multiplyOp() : operation("AAB::multiply") {}
    static multiplyOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new multiplyOp();
        op->registerInput(lhs, rhs);
        op->createValue(ctx->getType<lgf::variable>(), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }
    lgf::value* output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0)->represent()<<" * "<<inputValue(1)->represent();
        return p.dump();
    }
};

// ---------- inverseOp ----------
class inverseOp : public lgf::operation
{
    public:
    inverseOp() : operation("AAB::inverse") {}
    static inverseOp* build(lgf::LGFContext* ctx, lgf::value* input){
        auto op = new inverseOp();
        op->registerInput(input);
        op->createValue(input->getType(), "");
        return op;
    }
    lgf::value* input(){ return inputValue(0); }
    lgf::value* output(){ return outputValue(1); }
};

// ---------- quotientOp ----------
class quotientOp : public lgf::operation
{
    public:
    quotientOp() : operation("AAB::quotient"){}
    static quotientOp* build(lgf::LGFContext *ctx, lgf::value* x, lgf::value* y){
        auto op = new quotientOp();
        op->registerInput(x, y);
        op->createValue(x->getType(), "");
        return op;
    }
    lgf::value* numerator(){ return inputValue(0); }
    lgf::value* denominator(){ return inputValue(1); }
    lgf::value* output(){ return outputValue(1); }
};

class powerOp : public lgf::operation
{
    public:
    powerOp() : operation("AAB::power"){}
    static powerOp* build(lgf::LGFContext* ctx, lgf::value* x, lgf::value *y){
        auto op = new powerOp();
        op->registerInput(x, y);
        op->createValue(x->getType(), "");
        return op;
    }
    lgf::value* power(){ return inputValue(1); }
    lgf::value* x(){ return inputValue(0); }
    lgf::value* output() { return outputValue(1); }
};

class function1DOp: public lgf::operation {
    public:
    function1DOp(std::string name) : operation(name){}
    static function1DOp* build(lgf::LGFContext* ctx, lgf::value* x){
        auto op = new function1DOp("function1DOp");
        op->registerInput(x);
        op->createValue(x->getType(), "");
        return op;
    }
    lgf::value* x(){ return inputValue(0); }
    lgf::value* output(){ return outputValue(1); }
};

class funcSineOp : public function1DOp{
    public:
    funcSineOp() :  function1DOp("sine"){}
    static funcSineOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new funcSineOp();
        op->registerInput(x);
        op->createValue(x->getType(), "");
        return op;
    }
};

class funcCosOp : public function1DOp{
    public:
    funcCosOp(): function1DOp("cos"){}
    static funcCosOp* build (lgf::LGFContext* ctx, lgf::value* x){
        auto op = new funcCosOp();
        op->registerInput(x);
        op->createValue(x->getType(), "");
        return op;
    }
};

class derivativeOp : public operation {
    public:
    derivativeOp() : operation("derivative"){}
    static derivativeOp* build(lgf::LGFContext* ctx, lgf::value* func, value* var){
        auto op = new derivativeOp();
        op->registerInput(func, var);
        op->createValue(func->getType(), "");
        return op;
    }
    lgf::value* func(){ return inputValue(0); }
    lgf::value* output(){ return outputValue(1); }
};

}

#endif