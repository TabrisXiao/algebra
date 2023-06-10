#include "lgf/operation.h"

// ---------- addOp ----------
class addOp : public lgf::operation
{
  public:
  addOp(lgf::value& lhs, lgf::value& rhs){
    registerInput(lhs, rhs);
    createValue("variable", "output");
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- multiplyOp ----------
class multiplyOp : public lgf::operation
{
  public:
  multiplyOp(lgf::value& lhs, lgf::value& rhs){
    registerInput(lhs, rhs);
    createValue("variable", "output");
  }
  lgf::value& lhs(){ return inputValue(0); }
  lgf::value& rhs(){ return inputValue(1); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- defOp ----------
class defOp : public lgf::operation
{
  public:
  defOp(){
    createValue("variable", "output");
  }
  lgf::value& output(){ return outputValue(0); }
};
