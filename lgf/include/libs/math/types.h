#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "lgf/types.h"

namespace math{
class variable: public lgf::type_t {
  public:
  variable() { id="variable"; }
  static variable build(){
    variable obj;
    return obj;
  }
};

class natureNumber: public variable {
  public:
  natureNumber() { id="natureNumber"; }
  static natureNumber build(){
    natureNumber obj;
    return obj;
  }
};

class integer: public variable {
  public:
  integer() { id="integer"; }
  static integer build(){
    integer obj;
    return obj;
  }
};

class rationalNumber: public variable {
  public:
  rationalNumber() { id="rationalNumber"; }
  static rationalNumber build(){
    rationalNumber obj;
    return obj;
  }
};

class irrationalNumber: public variable {
  public:
  irrationalNumber() { id="irrationalNumber"; }
  static irrationalNumber build(){
    irrationalNumber obj;
    return obj;
  }
};

class realNumber: public variable {
  public:
  realNumber() { id="realNumber"; }
  static realNumber build(){
    realNumber obj;
    return obj;
  }
};

// class anyMatrix: public variable {
//   public:
//   anyMatrix() { id="anyMatrix"; }
//   static anyMatrix build(){
//     anyMatrix obj;
//     return obj;
//   }
// };

// class matrix: public anyMatrix {
//   public:
//   matrix() { id="matrix"; }
//   static matrix build(type_t elemType, int rowDim, int colDim){
//     matrix obj;
//     obj.elemType_=elemType;
//     obj.rowDim_=rowDim;
//     obj.colDim_=colDim;
//     return obj;
//   }
//   const type_t& elemType(){ return elemType_; }
//   const int& rowDim(){ return rowDim_; }
//   const int& colDim(){ return colDim_; }
//   type_t elemType_;
//   int rowDim_;
//   int colDim_;
// };

class matrix: public lgf::variable {
  public:
  matrix(lgf::variable elem_t) : elemType(elem_t) { id="matrix"; }
  static matrix build(lgf::variable t= lgf::variable()){
    return matrix(t);
  }
  lgf::variable elemType;
  int rowDim = -1, colDim = -1;
};

}
#endif
