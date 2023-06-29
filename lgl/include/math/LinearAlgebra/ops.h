#ifndef LINALG_OPS_H
#define LINALG_OPS_H
#include "types.h"
#include "lgf/operation.h"

namespace LinAlg{

// ---------- addOp ----------
class addOp : public lgf::operation
{
  public:
  addOp(math::matrix output_t, lgf::value& lhs, lgf::value& rhs){
    setSID("LinAlg::addOp");
    lhs.type_guard<math::matrix>();
    rhs.type_guard<math::matrix>();
    registerInput(lhs, rhs);
    createValue(output_t, "output");
  }
  static addOp* build(math::matrix output_t, lgf::value& lhs, lgf::value& rhs){
    auto op = new addOp(output_t, lhs, rhs);
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
  multiplyOp(math::matrix output_t, lgf::value& lhs, lgf::value& rhs){
    setSID("LinAlg::multiplyOp");
    lhs.type_guard<math::matrix>();
    rhs.type_guard<math::matrix>();
    registerInput(lhs, rhs);
    createValue(output_t, "output");
  }
  static multiplyOp* build(math::matrix output_t, lgf::value& lhs, lgf::value& rhs){
    auto op = new multiplyOp(output_t, lhs, rhs);
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
  inverseOp(math::matrix output_t, lgf::value& input){
    setSID("LinAlg::inverseOp");
    input.type_guard<math::matrix>();
    registerInput(input);
    createValue(output_t, "output");
  }
  static inverseOp* build(math::matrix output_t, lgf::value& input){
    auto op = new inverseOp(output_t, input);
    return op;
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- transpose ----------
class transpose : public lgf::operation
{
  public:
  transpose(math::matrix output_t, lgf::value& input){
    setSID("LinAlg::transpose");
    input.type_guard<math::matrix>();
    registerInput(input);
    createValue(output_t, "output");
  }
  static transpose* build(math::matrix output_t, lgf::value& input){
    auto op = new transpose(output_t, input);
    return op;
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- declareOp ----------
class declareOp : public lgf::operation
{
  public:
  declareOp(math::matrix output_t){
    setSID("LinAlg::declareOp");
    createValue(output_t, "output");
  }
  static declareOp* build(math::matrix output_t){
    auto op = new declareOp(output_t, );
    return op;
  }
  lgf::value& output(){ return outputValue(0); }
};

}
#endif
