#ifndef AAB_OPS_H
#define AAB_OPS_H
#include "types.h"
#include "lgf/operation.h"

namespace aab{

// ---------- addOp ----------
class addOp : public lgf::operation
{
  public:
  addOp(variable output_t, lgf::value& lhs, lgf::value& rhs){
    lhs.type_guard<variable>();
    rhs.type_guard<variable>();
    registerInput(lhs, rhs);
    createValue(output_t, "output");
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
  public:
  multiplyOp(variable output_t, lgf::value& lhs, lgf::value& rhs){
    lhs.type_guard<variable>();
    rhs.type_guard<variable>();
    registerInput(lhs, rhs);
    createValue(output_t, "output");
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- inverseOp ----------
class inverseOp : public lgf::operation
{
  public:
  inverseOp(variable output_t, lgf::value& input){
    input.type_guard<variable>();
    registerInput(input);
    createValue(output_t, "output");
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- defOp ----------
class defOp : public lgf::operation
{
  public:
  defOp(variable output_t){
    createValue(output_t, "output");
  }
  lgf::value& output(){ return outputValue(0); }
};

}
#endif
