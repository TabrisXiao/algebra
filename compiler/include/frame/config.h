
#ifndef FRAME_CONFIG_H_
#define FRAME_CONFIG_H_

#define SUCCESS 1
#define FAIL 0
#include "symbolTable.h"

namespace MC{
    static auto gTable = symbolTable::getInstance();

    // void init(){
    //     gTable->registerSymbol("unknown");
    // }

}

#endif