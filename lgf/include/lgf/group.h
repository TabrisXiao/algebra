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
    virtual resultCode rewrite( painter, operation *op) = 0;
};

template<typename groupType>
class groupRewriter : public rewriterBase{
    public: 
    groupRewriter() = default;
    virtual resultCode execute( painter rewriter,operation* op) override final{
        if(auto g = dynamic_cast<groupType*>(op))
        {
            auto sig = g->rewrite(rewriter, op);
            return sig;
        }
        return resultCode::pass();
    }
};

class normalizer : public group {
    public: 
    normalizer() = default;
};

class normalizationPass : public passBase {
    public: 
    normalizationPass() : passBase("normalization"){}
    virtual resultCode run(){
        painter p(getContext());
        addRewriter<groupRewriter<normalizer>>();
        applyRewriterOnce(p, getGraph());
        return applyRewriterOnce(p, getGraph());
        //return applyRewriterGreedy(p, getGraph());
    }
};

}

#endif