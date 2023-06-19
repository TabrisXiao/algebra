#include "lgf/operation.h"
#include "types.h"

// ---------- addOp ----------
class addOp : public lgf::operation
{
  public:
  addOp(lgf::type_t output_t, lgf::value& lhs, lgf::value& rhs){
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
  multiplyOp(lgf::type_t output_t, lgf::value& lhs, lgf::value& rhs){
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
  inverseOp(lgf::type_t output_t, lgf::value& input){
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
  defOp(lgf::type_t output_t){
    createValue(output_t, "output");
  }
  lgf::value& output(){ return outputValue(0); }
};
