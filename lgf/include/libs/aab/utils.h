
#ifndef MATH_AAB_UTILS_H
#define MATH_AAB_UTILS_H
#include "lgf/operation.h"
#include "libs/AAB/ops.h"

namespace lgf::AAB{
bool checkInverse(value* lhs, value* rhs){
    // check if lhs are inverse of rhs
    if(auto inv = lhs->getDefiningOp<AAB::inverseOp>()){
        if(inv->inputValue(0) == rhs) return true;
    }
    return false;
}

bool checkMutualInverse(value* lhs, value* rhs){
    return checkInverse(lhs, rhs) || checkInverse(rhs, lhs);
}
}

#endif // MATH_AAB_UTILS_H