#ifndef AAB_OPS_H
#define AAB_OPS_H
#include "types.h"
#include "lgf/operation.h"

namespace aab{

// ---------- addOp ----------
class addOp : public lgf::operation
{
  public:
  addOp(){}
  static addOp* build(math::variable output_t, lgf::value& lhs, lgf::value& rhs){
    lhs.type_guard<math::variable>();
    rhs.type_guard<math::variable>();
    auto op = new addOp();
    op->registerInput(lhs, rhs);
    op->setSID("aab::addOp");
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
  static inverseOp* build(math::variable output_t, lgf::value& input){
    input.type_guard<math::variable>();
    auto op = new inverseOp();
    op->registerInput(input);
    op->setSID("aab::inverseOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- defOp ----------
class defOp : public lgf::operation
{
  public:
  defOp(){}
  static defOp* build(math::variable output_t){
    auto op = new defOp();
    op->setSID("aab::defOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
  public:
  multiplyOp(){}
  static multiplyOp* build(math::variable output_t, lgf::value& lhs, lgf::value& rhs){
    lhs.type_guard<math::variable>();
    rhs.type_guard<math::variable>();
    auto op = new multiplyOp();
    op->registerInput(lhs, rhs);
    op->setSID("aab::multiplyOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

}
#endif
