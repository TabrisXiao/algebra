#ifndef ANALYSIS_OPS_H
#define ANALYSIS_OPS_H
#include "types.h"
#include "lgf/operation.h"

namespace analysis{

// ---------- mappingOp ----------
class mappingOp : public lgf::operation
{
  public:
  mappingOp(){}
  static mappingOp* build(math::variable output_t, lgf::value& x){
    x.type_guard<math::variable>();
    auto op = new mappingOp();
    op->registerInput(x);
    op->setSID("analysis::mappingOp");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- sinFunc ----------
class sinFunc : public lgf::operation
{
  public:
  sinFunc(){}
  static sinFunc* build(math::variable output_t, lgf::value& x){
    x.type_guard<math::variable>();
    auto op = new sinFunc();
    op->registerInput(x);
    op->setSID("analysis::sinFunc");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- cosFunc ----------
class cosFunc : public lgf::operation
{
  public:
  cosFunc(){}
  static cosFunc* build(math::variable output_t, lgf::value& x){
    x.type_guard<math::variable>();
    auto op = new cosFunc();
    op->registerInput(x);
    op->setSID("analysis::cosFunc");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- expFunc ----------
class expFunc : public lgf::operation
{
  public:
  expFunc(){}
  static expFunc* build(math::variable output_t, lgf::value& x){
    x.type_guard<math::variable>();
    auto op = new expFunc();
    op->registerInput(x);
    op->setSID("analysis::expFunc");
    op->createValue(output_t, "output");
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

}
#endif
