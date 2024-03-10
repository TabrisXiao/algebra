
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

bool mergeUnit(int i, operation* op){
    auto inputs = op->getInputs();
    auto unit_base_type = inputs[i]->getType<unitType>().getBaseType();
    if(i > 0){
        if(inputs[i-1]->getType() == unit_base_type){
            op->dropInputValue(i);
            return true;
        }
    }
    if(i < inputs.size()){
        if(inputs[i+1]->getType() == unit_base_type){
            op->dropInputValue(i);
            return true;
        }
    }
    return false;
}

bool mergeZero(int i, operation* op){
    auto inputs = op->getInputs();
    auto zero_base_type = inputs[i]->getType<zeroType>().getBaseType();
    if(i > 0){
        if(inputs[i-1]->getType() == zero_base_type){
            op->dropInputValue(i-1);
            return true;
        }
    }
    if(i < inputs.size()){
        if(inputs[i+1]->getType() == zero_base_type){
            op->dropInputValue(i+1);
            return true;
        }
    }
    return false;
}
}

#endif // MATH_AAB_UTILS_H