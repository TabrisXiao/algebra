#ifndef GROUP_H
#define GROUP_H
#include <memory.h>
#include "operation.h"
#include "painter.h"

namespace lgf{

class interfaceBase {
    public:
    interfaceBase () = default;
};

// group is used to classify operations based on general properties,
// usages, or whatever general ideas. And also provide interface methods
// to this group. 
template<typename concreteType>
class group : public operation{
    public: 
    group () = default;
    virtual ~group(){}

    // method to check if an op belongs to certain group.
    static bool isGroup(operation *op){
        if(auto gp = dynamic_cast<group<concreteType>>(op))
            return true;
        return false;
    }

    virtual bool normalize(painter&, operation *) {return 0;}
    template<typename groupTy>
    bool tryNormalize(painter & p){
        if(dynamic_cast<groupTy*>(this)){
            if(auto op = dynamic_cast<operation*>(this))
                return normalize(p, op);
        }
        return 0;
    }
};

}

#endif