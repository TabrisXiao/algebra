
#ifndef COMPILER_LGBUILDER_H
#define COMPILER_LGBUILDER_H
#include "lgf/lgf.h"
#include "compiler/ast.h"
#include "libs/math/aab/ops.h"
#include "context.h"

namespace lgf::compiler {

class LGTranslator {
    public: 
    LGTranslator() = default;
    void build(std::unique_ptr<moduleAST> & main){
        pnt.gotoGraph(&c);
        auto module = pnt.createOp<moduleOp>();
        pnt.gotoGraph(module);
        declareVariables(*(ctx->current_scope));
        translateModuleAST(main);
    }
    void translateModuleAST(std::unique_ptr<moduleAST> & main){
        for(auto & op : main->contents){
            auto kind = op->kind;
            switch(kind){
                case kind_binary:
                    convertBinaryOp(op);
            }
        }
    }
    void declareVariables(context::scope& scp){
        for(auto& it : scp.stbl.table){
            auto & entry = it.second;
            if(entry.category != "variable") continue;
            pnt.createOp<declOp>(math::variable());
        }
    }
    void convertBinaryOp(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<binaryAST*>(op.get());
        auto bop = ast->binaryOp;
        if(bop == "+"){
            //pnt.createOp<math::aab::addOp>();
        }
    }
    std::unique_ptr<moduleAST> main;
    painter pnt;
    canvas c;
    context *ctx;
};

}

#endif