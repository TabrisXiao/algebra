
#ifndef LGF_COMPILER_H
#define LGF_COMPILER_H

#include "operation.h"
#include "pass.h"
namespace lgf{
class compilerPrototype{
    public:
    compilerPrototype() = default;
    passManager* getManager(){ return &pm;}
    void compile(LGFContext* ctx, graph* g){
        pm.init(ctx, g);
        pm.run();
    }
    virtual void build_pipeline() = 0;
    passManager pm;
};
}// namespace lgf

#endif