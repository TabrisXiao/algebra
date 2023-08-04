
#ifndef COMPILER_LGBUILDER_H
#define COMPILER_LGBUILDER_H
#include "lgf/lgf.h"
#include "compiler/ast.h"
#include "libs/math/math.h"
#include "context.h"
#include "lgf/typeTable.h"

namespace lgf::compiler {

class LGTranslator {
    public: 
    LGTranslator() = default;
    void build(std::unique_ptr<moduleAST> & main){
        registerLGFTypes();
        lgf::math::registerTypes();
        pnt.gotoGraph(&c);
        auto module = pnt.paint<moduleOp>(ctx);
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
                ptr = getVarValue(op);
                break;
            case kind_number:
                ptr = declareConstant(op);
                break;
            case kind_return:
                ptr = translateReturnOp(op);
                break;
            case kind_funcDecl:
                ptr = translateFuncDef(op);
                break;
            case kind_funcCall:
                ptr = translateFuncCall(op);
                break;
            default:
                translateError("unknown op type: "+std::to_string(op->kind));
                return nullptr;
        }
        return ptr;
    }
    operation * translateFuncDef(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<funcDeclAST*>(op.get());
        // assuming all function return variable for now.
        auto funcOp = pnt.paint<funcDefineOp>(ctx, ast->funcID, ctx->getType<lgf::variable>());
        astctx->current_scope->findSymbolInfo(ast->funcID)->handle = funcOp->outputValue(1);
        for(auto i=0; i< ast->args.size(); i++){
            auto arg = dynamic_cast<varDeclAST*>(ast->args[i].get());
            auto argid = arg->id;
            auto type = parseType(arg->typeStr);
            funcOp->registerArg(type, "arg");
        }
        if(!ast->returnTypeStr.empty())
            funcOp->returnType = typeTable::get().parseTypeStr(ctx,ast->returnTypeStr);
        if(ast->contents.size()!=0){
            // parsing definition block;
        }
        return funcOp;
    }
    operation* translateFuncCall(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<funcCallAST*>(op.get());
        auto callee = astctx->current_scope->findSymbolInfo(ast->id)->handle;
        std::vector<value*> args; 
        args.reserve(5);
        for(auto i=0; i<ast->args.size(); i++){
            auto argOp = translateAST(ast->arg(i));
            args.push_back(argOp->outputValue(1));
        }
        auto fnCall = pnt.sketch<funcCallOp>(ctx, callee);
        fnCall->addArgs(args);
        pnt.addToGraph(fnCall);
        return fnCall;
    }
    operation* translateReturnOp(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<returnAST*>(op.get());
        operation *value = nullptr;
        if(ast->hasValue()) {
            value = translateAST(ast->value);
        }
        auto retOp = pnt.sketch<returnOp>(ctx);
        if(value){
            retOp->registerInput(value->outputValue(1));
        } 
        else{
            pnt.appendOp(retOp);
        }
        pnt.addToGraph(retOp);
        return retOp;
    }
    void translateError(std::string msg){
        std::cerr<<"Translation error: "<<msg<<std::endl;
    }
    void declareVariables(scope<ASTContext::idinfo>& scp){
        for(auto& it : scp.stbl.table){
            auto & entry = it.second;
            if(entry.category != "variable") continue;
            auto type = parseType(entry.type);
            auto op = pnt.paint<declOp>(ctx, type);
            op->output()->setSID("");
            it.second.handle=op->output();
        }
    }
    operation * getVarValue(std::unique_ptr<astBase>& op){
        auto var = dynamic_cast<varAST*>(op.get());
        auto id = var->id;
        auto owner = astctx->current_scope->findSymbolInfo(var->id)->handle->getDefiningOp();
        THROW_WHEN(owner == nullptr, "The value for the variable: "+id+" is not defined yet!");
        return owner;
    }
    operation* declareConstant(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<numberAST*>(op.get());
        cstDeclOp * lgfop;
        if(ast->isInt()){
            lgfop = pnt.paint<cstDeclOp>(ctx, int(ast->number));
        } else
            lgfop = pnt.paint<cstDeclOp>(ctx, ast->number);
        return lgfop;
    }
    operation* convertBinaryOp(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<binaryAST*>(op.get());
        auto lhs = translateAST(ast->lhs);
        auto rhs = translateAST(ast->rhs);
        auto bop = ast->binaryOp;
        if(bop == "+"){
            // all the operation converted from ast should contained only 1 output.
            return pnt.paint<math::aab::addOp>(ctx, lhs->outputValue(1), rhs->outputValue(1));
        }else if(bop == "-"){
            // all the operation converted from ast should contained only 1 output.
            return pnt.paint<math::aab::minusOp>(ctx, lhs->outputValue(1), rhs->outputValue(1));
        }else if(bop == "*"){
            return pnt.paint<math::aab::multiplyOp>(ctx, lhs->outputValue(1), rhs->outputValue(1));
        }else if(bop == "="){
            if(auto ptr = dynamic_cast<varAST*>(ast->lhs.get())){
                auto ret = pnt.paint<assignOp>(ctx, lhs->outputValue(1), rhs->outputValue(1));
                astctx->current_scope->findSymbolInfo(ptr->id)->handle=ret->outputValue(1);
                return ret;
            }else {
                translateError("lhs of assignment has to be a variable.");
            }
            
        }
        return nullptr;
    }
    type_t parseType(std::string typeStr){
        return typeTable::get().parseTypeStr(ctx, typeStr);
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