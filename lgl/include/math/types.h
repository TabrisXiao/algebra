#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "lgf/operation.h"

namespace math{
class variable: public lgf::type_t {
  public:
  variable() { id="variable"; }
  static variable build(){
    variable obj;
    return obj;
  }
};

class natureNumber: public lgf::type_t {
  public:
  natureNumber() { id="natureNumber"; }
  static natureNumber build(){
    natureNumber obj;
    return obj;
  }
};

class integer: public lgf::type_t {
  public:
  integer() { id="integer"; }
  static integer build(){
    integer obj;
    return obj;
  }
};

class rationalNumber: public lgf::type_t {
  public:
  rationalNumber() { id="rationalNumber"; }
  static rationalNumber build(){
    rationalNumber obj;
    return obj;
  }
};

class irrationalNumber: public lgf::type_t {
  public:
  irrationalNumber() { id="irrationalNumber"; }
  static irrationalNumber build(){
    irrationalNumber obj;
    return obj;
  }
};

class realNumber: public lgf::type_t {
  public:
  realNumber() { id="realNumber"; }
  static realNumber build(){
    realNumber obj;
    return obj;
  }
};

class anyMatrix: public lgf::type_t {
  public:
  anyMatrix() { id="anyMatrix"; }
  static anyMatrix build(){
    anyMatrix obj;
    return obj;
  }
};

class matrix: public anyMatrix {
  public:
  matrix() { id="matrix"; }
  static matrix build(type_t elemType, int rowDim, int colDim){
    matrix obj;
    obj.elemType_=elemType;
    obj.rowDim_=rowDim;
    obj.colDim_=colDim;
    return obj;
  }
  const type_t& elemType(){ return elemType_; }
  const int& rowDim(){ return rowDim_; }
  const int& colDim(){ return colDim_; }
  type_t elemType_;
  int rowDim_;
  int colDim_;
};


}
#endif
