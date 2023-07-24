
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
        auto module = pnt.createOp<moduleOp>(ctx);
        pnt.gotoGraph(module);
        declareVariables(*(astctx->current_scope));
        translateModuleAST(main);
        c.assignID(0);
    }
    void translateModuleAST(std::unique_ptr<moduleAST> & main){
        transplateASTScope(main->contents);
    }
    void transplateASTScope(std::vector<std::unique_ptr<astBase>>& contents){
        for(auto & op : contents){
            translateAST(op);
        }
    }
    operation* translateAST(std::unique_ptr<astBase>& op){
        auto kind = op->kind;
        operation* ptr= nullptr;
        switch(kind){
            case kind_binary:
                ptr = convertBinaryOp(op);
                break;
            case kind_variable:
                ptr = getVarDeclOp(op);
                break;
            case kind_number:
                ptr = declareConstant(op);
                break;
            case kind_return:
                ptr = translateReturnOp(op);
            case kind_funcDecl:
                ptr = translateFuncDef(op);
            default:
                translateError("unknown op type: "+std::to_string(op->kind));
                return nullptr;
        }
        return ptr;
    }
    operation * translateFuncDef(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<funcDeclAST*>(op.get());
        // assuming all function return variable for now.
        auto funcOp = pnt.createOp<funcDefineOp>(ctx, ast->funcID, ctx->getType<lgf::variable>());
        for(auto i=0; i< ast->args.size(); i++){
            auto arg = dynamic_cast<varAST*>(ast->args[i].get());
            auto argid = arg->id;
            funcOp->registerArg(ctx->getType<lgf::variable>(), "arg");
        }
        if(ast->contents.size()!=0){
            // parsing definition block;
        }
        return funcOp;
    }
    operation* translateReturnOp(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<returnAST*>(op.get());
        operation *value = nullptr;
        if(ast->hasValue()) {
            value = translateAST(ast->value);
        }
        auto retOp = pnt.createOp<returnOp>(ctx);
        if(value) retOp->registerInput(value->outputValue(0));
        else pnt.appendOp(retOp);
        return retOp;
    }
    void translateError(std::string msg){
        std::cerr<<"Translation error: "<<msg<<std::endl;
    }
    void declareVariables(scope<ASTContext::idinfo>& scp){
        for(auto& it : scp.stbl.table){
            auto & entry = it.second;
            if(entry.category != "variable") continue;
            auto op = pnt.createOp<declOp>(ctx, ctx->getType<variable>());
            op->output()->setSID("");
            it.second.ptr=op;
        }
    }
    operation * getVarDeclOp(std::unique_ptr<astBase>& op){
        auto var = dynamic_cast<varAST*>(op.get());
        auto id = var->id;
        auto defop = astctx->current_scope->findSymbolInfo(var->id)->ptr;
        THROW_WHEN(defop == nullptr, "The value for the variable: "+id+" is not defined yet!");
        return defop;
    }
    operation* declareConstant(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<numberAST*>(op.get());
        cstDeclOp * lgfop;
        if(ast->isInt()){
            lgfop = pnt.createOp<cstDeclOp>(ctx, int(ast->number));
        } else
            lgfop = pnt.createOp<cstDeclOp>(ctx, ast->number);
        return lgfop;
    }
    operation* convertBinaryOp(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<binaryAST*>(op.get());
        auto lhs = translateAST(ast->lhs);
        auto rhs = translateAST(ast->rhs);
        auto bop = ast->binaryOp;
        if(bop == "+"){
            // all the operation converted from ast should contained only 1 output.
            return pnt.createOp<math::aab::addOp>(ctx, ctx->getType<variable>(), lhs->outputValue(1), rhs->outputValue(1));
        }else if(bop == "-"){
            // all the operation converted from ast should contained only 1 output.
            return pnt.createOp<math::aab::minusOp>(ctx, ctx->getType<variable>(), lhs->outputValue(1), rhs->outputValue(1));
        }else if(bop == "*"){
            return pnt.createOp<math::aab::multiplyOp>(ctx, ctx->getType<variable>(), lhs->outputValue(1), rhs->outputValue(1));
        }else if(bop == "="){
            return pnt.createOp<assignOp>(ctx, ctx->getType<variable>(), lhs->outputValue(1), rhs->outputValue(1));
        }
        return nullptr;
    }
    std::unique_ptr<moduleAST> main;
    painter pnt;
    canvas c;
    ASTContext *astctx;
    LGFContext context;
    LGFContext *ctx = &context;
};

}

#endif