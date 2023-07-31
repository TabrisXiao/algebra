
#ifndef MATH_AAB_OPS_H
#define MATH_AAB_OPS_H
#include "lgf/types.h"
#include "math/types.h"
#include "lgf/operation.h"

namespace math::aab{

// ---------- addOp ----------
class addOp : public lgf::operation
{
    public:
    addOp(){}
    static addOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        auto op = new addOp();
        lhs->type_guard<math::tensor>();
        rhs->type_guard<math::tensor>();
        op->registerInput(lhs, rhs);
        op->setSID("linearAlg::add");
        op->createValue(lhs->inferType(rhs), "");
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
class minusOp : public lgf::operation
{
    public:
    minusOp(){}
    static minusOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        lhs->type_guard<math::tensor>();
        rhs->type_guard<math::tensor>();
        auto op = new minusOp();
        op->registerInput(lhs, rhs);
        op->setSID("linearAlg::minus");
        op->createValue(lhs->inferType(rhs), "");
        return op;
    }
    lgf::value* lhs(){ return inputValue(0); }
    lgf::value* rhs(){ return inputValue(1); }
    lgf::value* output(){ return outputValue(1); }
    virtual std::string represent(){
        printer p;
        p<<representOutputs()<<" = "<<getSID() <<" : "<<inputValue(0)->represent()<<" - "<<inputValue(1)->represent();
        return p.dump();
    }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
    public:
    multiplyOp(){}
    static multiplyOp* build(lgf::LGFContext* ctx, lgf::value* lhs, lgf::value* rhs){
        lhs->type_guard<math::tensor>();
        rhs->type_guard<math::tensor>();
        auto op = new multiplyOp();
        op->registerInput(lhs, rhs);
        op->setSID("linearAlg::multiply");
        op->createValue(lhs->inferType(rhs), "");
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
    inverseOp(){}
    static inverseOp* build(lgf::LGFContext* ctx, lgf::value* input){
        input->type_guard<math::tensor>();
        auto op = new inverseOp();
        op->registerInput(input);
        op->setSID("linearAlg::inverse");
        op->createValue(input->getType(), "");
        return op;
    }
    lgf::value* input(){ return inputValue(0); }
    lgf::value* output(){ return outputValue(0); }
};

class determinant : public lgf::operation {
    public:
    determinant () = default;
    static determinant* build(lgf::LGFContext*, lgf::value *input){
        input->type_guard<math::tensor>();
        auto op = new determinant();
        op->setSID("linearAlg::determinant");
        op->createValue(input->getType<math::tensor>().getElemType(), "");
        return op;
    }
};

}

#endif