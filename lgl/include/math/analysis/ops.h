#ifndef ANALYSIS_OPS_H
#define ANALYSIS_OPS_H
#include "types.h"
#include "lgf/operation.h"

namespace analysis{

// ---------- mappingOp ----------
class mappingOp : public lgf::operation
{
  public:
  mappingOp(math::variable output_t, lgf::value& x){
    setSID("analysis::mappingOp");
    x.type_guard<math::variable>();
    registerInput(x);
    createValue(output_t, "output");
  }
  static mappingOp* build(math::variable output_t, lgf::value& x){
    auto op = new mappingOp(output_t, x);
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- sinFunc ----------
class sinFunc : public lgf::operation
{
  public:
  sinFunc(math::variable output_t, lgf::value& x){
    setSID("analysis::sinFunc");
    x.type_guard<math::variable>();
    registerInput(x);
    createValue(output_t, "output");
  }
  static sinFunc* build(math::variable output_t, lgf::value& x){
    auto op = new sinFunc(output_t, x);
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- cosFunc ----------
class cosFunc : public lgf::operation
{
  public:
  cosFunc(math::variable output_t, lgf::value& x){
    setSID("analysis::cosFunc");
    x.type_guard<math::variable>();
    registerInput(x);
    createValue(output_t, "output");
  }
  static cosFunc* build(math::variable output_t, lgf::value& x){
    auto op = new cosFunc(output_t, x);
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- expFunc ----------
class expFunc : public lgf::operation
{
  public:
  expFunc(math::variable output_t, lgf::value& x){
    setSID("analysis::expFunc");
    x.type_guard<math::variable>();
    registerInput(x);
    createValue(output_t, "output");
  }
  static expFunc* build(math::variable output_t, lgf::value& x){
    auto op = new expFunc(output_t, x);
    return op;
  }
  lgf::value& x(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

}
#endif
