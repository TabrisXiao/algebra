#ifndef OPINTERFACE_H
#define OPINTERFACE_H
#include <memory.h>
#include "operation.h"
#include "aog.h"

namespace aog{
// a wrapper for containning the traits of operations
template<typename concreteType>
class opGroup{
    public: 
    opGroup () = default;
    virtual ~opGroup(){}
    static std::unique_ptr<concreteType> getRewriter(){
        return std::make_unique<concreteType>();
    }
};

}

#endif