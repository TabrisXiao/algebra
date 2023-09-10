#ifndef LGF_GROUP_H
#define LGF_GROUP_H
#include <memory.h>
#include "operation.h"
#include "painter.h"
#include "pass.h"

namespace lgf{

class group {
    public:
    group() = default;
    virtual bool rewrite( painter &, operation *op) = 0;
};

template<typename groupType>
class groupRewriter : public rewriterBase{
    public: 
    groupRewriter() = default;
    virtual int execute( painter & rewriter,operation* op) override final{
        // rewrite return value: 
        // 1 rewrite happen
        // 0 rewrite failed or not matched
        if(auto g = dynamic_cast<groupType*>(op))
        {
            auto sig = g->rewrite(rewriter, op);
            return int(sig);
        }
        return 0;
    }
};

class normalizer : public group {
    public: 
    normalizer() = default;
};

class normalizationPass : public passBase {
    public: 
    normalizationPass() : passBase("normalization"){}
    virtual bool run(){
        painter p(getContext());
        addRewriter<groupRewriter<normalizer>>();
        return applyRewriterGreedy(p, getGraph());
    }
};

}

#endif