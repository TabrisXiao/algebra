#include "lgf/operation.h"
#include "types.h"

// ---------- mappingOp ----------
class mappingOp : public lgf::operation
{
  public:
  mappingOp(variable output_t, lgf::value& input){
    input.type_guard<variable>();
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
  derivativeOp(variable derivative_t, lgf::value& var, lgf::value& func){
    var.type_guard<variable>();
    func.type_guard<variable>();
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
  sinFunc(variable output_t, lgf::value& var){
    var.type_guard<variable>();
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
  cosFunc(variable output_t, lgf::value& var){
    var.type_guard<variable>();
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
  expFunc(variable output_t, lgf::value& var){
    var.type_guard<variable>();
    registerInput(var);
    createValue(output_t, "output");
  }
  lgf::value& var(){ return inputValue(0); }
  lgf::value& output(){ return outputValue(0); }
};
