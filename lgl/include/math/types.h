#ifndef MATH_TYPES_H
#define MATH_TYPES_H
#include "lgf/operation.h"

namespace math{
class variable: public lgf::type_t {
    public:
    variable(){}
};

class integer: public lgf::type_t {
    public:
    integer(){}
};

class natureNumber: public lgf::type_t {
    public:
    natureNumber(){}
};

class rationalNumber: public lgf::type_t {
    public:
    rationalNumber(){}
};

class irrationalNumber: public lgf::type_t {
    public:
    irrationalNumber(){}
};

class realNumber: public lgf::type_t {
    public:
    realNumber(){}
};

class matrix: public lgf::type_t {
    public:
    matrix(type_t elemType, int rowDim, int colDim)
    : elemType_(elemType)
    , rowDim_(rowDim)
    , colDim_(colDim)
    {}
    const type_t& elemType(){ return elemType_; }
    const int& rowDim(){ return rowDim_; }
    const int& colDim(){ return colDim_; }
    type_t elemType_;
    int rowDim_;
    int colDim_;
};


}
#endif
