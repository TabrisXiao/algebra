
#ifndef COMPILER_LGFTRANSLATOR_H
#define COMPILER_LGFTRANSLATOR_H
#include "lgf/lgf.h"
#include "compiler/ast.h"
#include "libs/math.h"
#include "ASTContext.h"
#include "lgf/typeTable.h"

namespace lgf::compiler {

class LGTranslator {
    public: 
    LGTranslator() = default;
    void build(programAST* program){
        TRACE_LOG;
        astctx = program->getContext();
        astctx->resetModulePtr();
        lgf::math::registerTypes();
        pnt.gotoGraph(&c);
        for(auto & moduleast : program->modules){
            transplateASTModule(moduleast.get());
        }
        c.assignID(0);
    }
    void printModuleList(){
        std::cout<<"map ("<<astctx->module->getData()->name<<") size "<<astctx->module->table.size()<<", list: ";
        for(auto & pair : astctx->module->table){
            std::cout<<pair.first<<", ";
        }
        std::cout<<"\n";
    }
    void transplateASTModule(moduleAST* ast){
        TRACE_LOG;
        auto module = pnt.paint<moduleOp>(ctx, ast->name);
        pnt.gotoGraph(module);
        bool isMain = ast->name=="main";
        if(!isMain) astctx->moveToSubmodule(ast->name);
        if(option::get().log_lv_trace){
            std::cout<<"[--trace--]: ";
            printModuleList();
        } 
        astctx->module->getData()->ref = module->output();
        declareVariables(astctx->module->getData()->ids);
        translateASTBlock(ast->contents);
        if(!isMain) pnt.gotoParentGraph();
        astctx->moveToParentModule();
    }
    void translateASTBlock(std::vector<std::unique_ptr<astBase>>& contents){
        for(auto & op : contents){
            translateAST(op);
        }
    }
    value* translateAST(std::unique_ptr<astBase>& op){
        auto kind = op->kind;
        value* ptr= nullptr;
        switch(kind){
            case kind_access:
                ptr = translateAccessAST(op);
                break;
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

    value * translateAccessAST(std::unique_ptr<astBase>& op){
        TRACE_LOG;
        auto ast = dynamic_cast<accessAST*>(op.get());
        varAST* lhs = nullptr;
        accessAST * last= nullptr;
        auto curASTModuleLoc = astctx->module;
        while(ast){
            lhs = dynamic_cast<varAST*>(ast->lhs.get());
            auto id = lhs->id;
            if( auto idinfo = astctx->findSymbolInfoInCurrentModule(id) ){
            }else {
                astctx->moveToSubmodule(id);
            }
            last = ast;
            ast = dynamic_cast<accessAST*>(ast->rhs.get());
        }

        // if the final access object is a function
        if(auto fc = dynamic_cast<funcCallAST*>(last->rhs.get())){
            value* callee = nullptr;
            if(auto info = astctx->findSymbolInfoInCurrentModule(fc->id)){
                callee = info->handle;
                if(!callee) translateError("Function name is unknown: "+fc->id);
                auto op = pnt.sketch<funcCallOp>(ctx, callee);
                astctx->module = curASTModuleLoc;
                return convertFunCall(fc, op);
            }
            translateError("id name is unknown: "+fc->id);
        }
        translateError("Unknown access type");
        // auto mname = ast->getParentModuleName();
        // auto fc = ast->getEndAST();
        // value* callee = nullptr;
        // if(auto info = astctx->findSymbol(fc->id, mname)){
        //     callee = info->handle;
        //     if(!callee) translateError("Function name is unknown: "+fc->id);
        //     auto op = pnt.sketch<funcCallOp>(ctx, callee);
        //     return convertFunCall(fc, op);
        // } 
        // it means translation fails
        
        return nullptr;
    }
    value * translateFuncDef(std::unique_ptr<astBase>& op){
        TRACE_LOG;
        auto ast = dynamic_cast<funcDeclAST*>(op.get());
        // assuming all function return variable for now.
        auto funcOp = pnt.paint<funcDefineOp>(ctx, ast->funcID, ctx->getType<lgf::variable>());
        astctx->findSymbolInfoInCurrentModule(ast->funcID)->handle = funcOp->getCallee();
        astctx->createSubmoduleAndEnter(ast->funcID);
        for(auto i=0; i< ast->args.size(); i++){
            auto arg = dynamic_cast<varDeclAST*>(ast->args[i].get());
            auto argid = arg->id;
            auto type = parseType(arg->typeStr);
            funcOp->registerArg(type, "arg");
            astctx->addSymbolInfoToCurrentScope(argid, {"arg", arg->loc, arg->typeStr, funcOp->argument(i)});
        }
        if(!ast->returnTypeStr.empty())
            funcOp->returnType = typeTable::get().parseTypeStr(ctx,ast->returnTypeStr);
        
        if(!ast->isAbstract){
            funcOp->isAbstract = 0;
            pnt.gotoGraph(funcOp);
            declareVariables(astctx->module->getData()->ids);
            translateASTBlock(ast->contents);
            pnt.gotoParentGraph();
        }
        astctx->deleteCurrentModule();
        return funcOp->getCallee();
    }
    value* translateFuncCall(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<funcCallAST*>(op.get());
        auto callee = astctx->findSymbolInfoInCurrentModule(ast->id)->handle;
        auto fnCall = pnt.sketch<funcCallOp>(ctx, callee);
        return convertFunCall(ast, fnCall);
    }
    value* convertFunCall(funcCallAST* ast, funcCallOp *op){
        TRACE_LOG;
        //std::vector<value*> args; 
        for(auto i=0; i<ast->args.size(); i++){
            auto arg = translateAST(ast->arg(i));
            //args.push_back(arg);
            op->registerInput(arg);
        }
        //op->addArgs(args);
        pnt.addToGraph(op);
        return op->returnValue();
    }
    value* translateReturnOp(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<returnAST*>(op.get());
        value *val = nullptr;
        if(ast->hasValue()) {
            val = translateAST(ast->value);
        }
        auto retOp = pnt.sketch<returnOp>(ctx);
        value* ret = nullptr;
        if(val){
            retOp->registerInput(val);
        } 
        else{
            pnt.appendOp(retOp);
        }
        pnt.addToGraph(retOp);
        
        return ret;
    }
    void translateError(std::string msg){
        std::cerr<<"Translation error: "<<msg<<std::endl;
    }
    void declareVariables(symbolTable<idinfo>& idtbl){
        for(auto& it : idtbl.table){
            auto & entry = it.second;
            if(entry.category == "var"){
                auto type = parseType(entry.type);
                auto op = pnt.paint<declOp>(ctx, type);
                op->output()->setSID("");
                it.second.handle=op->output();
            }
        }
    }
    value * getVarValue(std::unique_ptr<astBase>& op){
        auto var = dynamic_cast<varAST*>(op.get());
        auto id = var->id;
        auto info = astctx->findSymbolInfoInCurrentModule(var->id);
        THROW_WHEN(info == nullptr, "The value for the variable: "+id+" is not defined yet!");
        return info->handle;
    }
    value* declareConstant(std::unique_ptr<astBase>& op){
        auto ast = dynamic_cast<numberAST*>(op.get());
        cstDeclOp * lgfop;
        if(ast->isInt()){
            lgfop = pnt.paint<cstDeclOp>(ctx, int(ast->number));
        } else
            lgfop = pnt.paint<cstDeclOp>(ctx, ast->number);
        return lgfop->output();;
    }
    value* convertBinaryOp(std::unique_ptr<astBase>& op){
        TRACE_LOG;
        auto ast = dynamic_cast<binaryAST*>(op.get());
        auto lhs = translateAST(ast->lhs);
        auto rhs = translateAST(ast->rhs);
        auto bop = ast->binaryOp;
        if(bop == "+"){
            // all the operation converted from ast should contained only 1 output.
            return pnt.paint<math::aab::addOp>(ctx, lhs, rhs)->output();
        }else if(bop == "-"){
            // all the operation converted from ast should contained only 1 output.
            return pnt.paint<math::aab::minusOp>(ctx, lhs, rhs)->output();
        }else if(bop == "*"){
            return pnt.paint<math::aab::multiplyOp>(ctx, lhs, rhs)->output();
        }else if(bop == "="){
            if(auto ptr = dynamic_cast<varAST*>(ast->lhs.get())){
                auto ret = pnt.paint<assignOp>(ctx, lhs, rhs)->output();
                astctx->findSymbolInfoInCurrentModule(ptr->id)->handle=ret;
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
    ASTContext *astctx=nullptr;
    LGFContext *ctx = nullptr;
    nestedSymbolicTable<moduleInfo>* temp_ptr = astctx;
};

}

#endif