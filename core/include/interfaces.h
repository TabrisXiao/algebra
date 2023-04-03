
#ifndef BUILTIN_INTERFACES_H
#define BUILTIN_INTERFACES_H

#include "aog.h"
#include "opInterface.h"
#include "pass.h"
namespace aog{
// for operation that inputs are commutable 
class commutable : public opGroup<commutable>, public rewriter<opGroup<commutable>> {
    public:
    commutable () = default;
    virtual bool rewrite(opRewriter &rewriter, opGroup<commutable> *origOp) override{
        auto op = dynamic_cast<operation*>(origOp);
        std::vector<element*> & vec = op->getInputs();
        std::sort(vec.begin(), vec.end(), [](element* a, element* b) {return a < b; });
        return 0;
    }
};

// this type of pass is used specific for trait normalization if it has
class normalizationPass : public passBase {
    public:
    normalizationPass() : passBase("normalization pass"){}
    bool run() final{
        graphModifier gm;
        gm.addRewriter<commutable>();
        auto reg = getRegion();
        gm.walkApplyOnce(reg);
        return 0;
    }
};

void createNormalizationPass(passManager &pm){
    pm.addPass(std::make_unique<normalizationPass>());
}
}
#endif