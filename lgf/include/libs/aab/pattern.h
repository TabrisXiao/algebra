#ifndef LIB_AAB_PATTERN_H
#define LIB_AAB_PATTERN_H
#include "lgf/utils.h"
namespace lgf{
using namespace utils;

template<typename opType1, typename opType2>
logicResult rDistributivePattern(operation *root){
    // the pattern is
    //    opType1
    //    /    \
    // value   opType2
    auto op1 = dynamic_cast<opType1*>(root);
    opType2* op2 = root->inputValue(1)->getDefiningOp<opType2>();
    if(op1 && op2)
        return logicResult::success();
    return logicResult::fail();
}
template<typename opType1, typename opType2>
logicResult lDistributivePattern(operation *root){
    // the pattern is
    //    opType1
    //    /    \
    // opType2   value
    auto op1 = dynamic_cast<opType1*>(root);
    opType2* op2 = root->inputValue(0)->getDefiningOp<opType2>();
    if(op1 && op2)
        return logicResult::success();
    return logicResult::fail();
}


}

#endif