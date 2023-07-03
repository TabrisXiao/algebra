#ifndef LINALG_OPS_H
#define LINALG_OPS_H
#include "types.h"
#include "lgf/operation.h"

namespace LinAlg{

// ---------- addOp ----------
class addOp : public lgf::operation
{
  public:
  addOp(){}
  static addOp* build(math::matrix output_t, lgf::value& lhs, lgf::value& rhs){
    lhs.type_guard<math::matrix>();
    rhs.type_guard<math::matrix>();
    auto op = new addOp();
    op->registerInput(lhs, rhs);
    op->setSID("LinAlg::addOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
  public:
  multiplyOp(){}
  static multiplyOp* build(math::matrix output_t, lgf::value& lhs, lgf::value& rhs){
    lhs.type_guard<math::matrix>();
    rhs.type_guard<math::matrix>();
    auto op = new multiplyOp();
    op->registerInput(lhs, rhs);
    op->setSID("LinAlg::multiplyOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- inverseOp ----------
class inverseOp : public lgf::operation
{
  public:
  inverseOp(){}
  static inverseOp* build(math::matrix output_t, lgf::value& input){
    input.type_guard<math::matrix>();
    auto op = new inverseOp();
    op->registerInput(input);
    op->setSID("LinAlg::inverseOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- transpose ----------
class transpose : public lgf::operation
{
  public:
  transpose(){}
  static transpose* build(math::matrix output_t, lgf::value& input){
    input.type_guard<math::matrix>();
    auto op = new transpose();
    op->registerInput(input);
    op->setSID("LinAlg::transpose");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- declareOp ----------
class declareOp : public lgf::operation
{
  public:
  declareOp(){}
  static declareOp* build(math::matrix output_t){
    auto op = new declareOp();
    op->setSID("LinAlg::declareOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& output(){ return outputValue(0); }
};

}
#endif
