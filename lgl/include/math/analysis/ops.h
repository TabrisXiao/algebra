#include "lgf/operation.h"
#include "types.h"

// ---------- mappingOp ----------
class mappingOp : public lgf::operation
{
  public:
  mappingOp(lgf::type_t output_t, lgf::value& input){
    registerInput(input);
    createValue(output_t, "output");
  }
  lgf::value& input(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- derivativeOp ----------
class derivativeOp : public lgf::operation
{
  public:
  derivativeOp(lgf::type_t derivative_t, lgf::value& var, lgf::value& func){
    registerInput(var, func);
    createValue(derivative_t, "derivative");
  }
  lgf::value& var(){ return inputValue(0); }
  lgf::value& func(){ return inputValue(1); }
  lgf::value& derivative(){ return outputValue(0); }
};

// ---------- sinFunc ----------
class sinFunc : public lgf::operation
{
  public:
  sinFunc(lgf::type_t output_t, lgf::value& var){
    registerInput(var);
    createValue(output_t, "output");
  }
  lgf::value& var(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- cosFunc ----------
class cosFunc : public lgf::operation
{
  public:
  cosFunc(lgf::type_t output_t, lgf::value& var){
    registerInput(var);
    createValue(output_t, "output");
  }
  lgf::value& var(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};

// ---------- expFunc ----------
class expFunc : public lgf::operation
{
  public:
  expFunc(lgf::type_t output_t, lgf::value& var){
    registerInput(var);
    createValue(output_t, "output");
  }
  lgf::value& var(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};
