
#ifndef MATH_AAB_OPS_H
#define MATH_AAB_OPS_H
#include "libs/math/types.h"
#include "lgf/operation.h"

namespace math::aab{

// ---------- addOp ----------
class addOp : public lgf::operation
{
    public:
    addOp(){}
    static addOp* build(math::variable output_type, lgf::value& lhs, lgf::value& rhs){
        auto op = new addOp();
        op->registerInput(lhs, rhs);
        op->setSID("aab::add");
        op->createValue(output_type, "");
        return op;
    }
    lgf::value& lhs(){ return inputValue(0); }
    lgf::value& rhs(){ return inputValue(1); }
    lgf::value& output(){ return outputValue(0); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0).represent()<<" + "<<inputValue(1).represent();
        return p.dump();
    }
};


// ---------- minusOp ----------
class minusOp : public lgf::operation
{
    public:
    minusOp(){}
    static minusOp* build(math::variable output_type, lgf::value& lhs, lgf::value& rhs){
        auto op = new minusOp();
        op->registerInput(lhs, rhs);
        op->setSID("aab::minus");
        op->createValue(output_type, "");
        return op;
    }
    lgf::value& lhs(){ return inputValue(0); }
    lgf::value& rhs(){ return inputValue(1); }
    lgf::value& output(){ return outputValue(0); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0).represent()<<" - "<<inputValue(1).represent();
        return p.dump();
    }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
    public:
    multiplyOp(){}
    static multiplyOp* build(math::variable output_type, lgf::value& lhs, lgf::value& rhs){
        auto op = new multiplyOp();
        op->registerInput(lhs, rhs);
        op->setSID("aab::multiply");
        op->createValue(output_type, "");
        return op;
    }
    lgf::value& lhs(){ return inputValue(0); }
    lgf::value& rhs(){ return inputValue(1); }
    lgf::value& output(){ return outputValue(0); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0).represent()<<" * "<<inputValue(1).represent();
        return p.dump();
    }
};

// ---------- inverseOp ----------
class inverseOp : public lgf::operation
{
    public:
    inverseOp(){}
    static inverseOp* build(math::variable output_type, lgf::value& input){
        auto op = new inverseOp();
        op->registerInput(input);
        op->setSID("aab::inverse");
        op->createValue(output_type, "");
        return op;
    }
    lgf::value& input(){ return inputValue(0); }
    lgf::value& output(){ return outputValue(0); }
};

}

#endif