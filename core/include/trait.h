#ifndef OPINTERFACE_H
#define OPINTERFACE_H
#include <memory.h>
#include "operation.h"
#include "aog.h"

namespace aog{
// a wrapper for containning the traits of operations
template<typename concreteTrait>
class trait{
    public: 
    trait () = default;
    virtual ~trait(){}
    static std::unique_ptr<concreteTrait> getNormalize(){
        return std::make_unique<concreteTrait>();
    }
};

// for operation that inputs are commutable 
class commutable : public rewriter<trait<commutable>> {
    public:
    commutable () = default;
    virtual bool rewrite(opRewriter &rewriter, trait<commutable> *origOp) override{
        auto op = dynamic_cast<operation*>(origOp);
        std::vector<element*> & vec = op->getInputs();
        std::sort(vec.begin(), vec.end(), [](element* a, element* b) { return a < b; });
        return 1;
    }
};

}

#endif