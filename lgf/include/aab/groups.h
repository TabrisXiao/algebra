#ifndef LGGROUPS_H_
#define LGGROUPS_H_
#include "lgf/operation.h"
#include "lgf/group.h"
namespace lgf
{
class commutable : public interfaceBase {
    public:
    commutable() = default;

    // Reorder the inputs in based on the address of defining op.
    // So this order is random but uniform. 
    void randomReorder(operation *op) {
        auto& inputs = op->getInputRefs();
        std::sort(inputs.begin(), inputs.end(), 
        [](valueRef& a, valueRef &b){
            return a.getDefiningOp() > b.getDefiningOp();
        });
    }
    void commute(operation * op, int n, int m){
        auto& inputs = op->getInputRefs();
        std::swap(inputs[n], inputs[m]);
    }
};

class associative : public interfaceBase{
    public:
    associative() = default;
    // the definition of the associative direction
    // (using a multiply as an example):
    // forward:  (a*b)*c = a*(b*c) 
    // backward: a*(b*c) = (a*b)*c

    // find the pattern that associative law could apply to
    template <typename opTy>
    std::vector<operation*> findPattern(opTy* op){
        // todo: add implementation
        std::vector<operation*> res;
        return res;
    }
};
}
#endif