
#ifndef COMPILER_PARSER_H
#define COMPILER_PARSER_H
#include "fileIO.h"
#include "lexer.h"
#include "ast.h"
#include "lgf/symbolicTable.h"
#include <memory>
#include <map>
#include <functional>
#include "lgf/operation.h"
#include "ASTContext.h"
#include <stack>
#include <fstream>

namespace lgf::compiler{
#define TRACE_FUNC_CALL \
    std::cout << "Calling function: " << __func__ << std::endl;

class parser{
    public:
    ~parser(){}
    parser(fileIO *io_) : io(io_) { }
    programAST* parseMainFile(fs::path path, programAST* ast_){
        program = ast_;
        ctx = ast_->getContext();
        lx.loadBuffer(path);
        lx.getNextToken();
        if(lx.getCurToken()==tok_module)
            lx.readNextLine();
        auto m = parseModule("main");
        program->addModule(std::move(m));
        registerID(program);
        return program;
    }
    programAST* parseModuleFile(fs::path path, programAST* ast){
        program = ast;
        ctx = ast->getContext();
        lx.loadBuffer(path);
        lx.getNextToken();
        auto id = path.stem().string();
        auto m = parseModule(id);
        program->addModule(std::move(m));
        return program;
    }
    void parseError(const char * msg){
        std::cerr<<lx.getLoc().string()<<": Error: "<<msg;
        std::exit(EXIT_FAILURE);
    }
    void parseError(std::string msg){
        parseError(msg.c_str());
    }

    std::unique_ptr<moduleAST> parseModule(std::string name){
        auto module = std::make_unique<moduleAST>(lx.getLoc(), ctx->module_id++);
        module->name = name;
        std::string id;
        while(true){
            std::unique_ptr<astBase> record = nullptr;
            switch(lx.getCurToken()){
                case tok_eof:
                    break;
                case tok_comment:
                    lx.getNextLine();
                    lx.getNextToken();
                    continue;
                case tok_module:
                    lx.consume(tok_module);
                    id = lx.parseIdentifier();
                    lx.consume(token('('));
                    record = parseModule(id);
                    lx.consume(token(')'));
                    break;
                case tok_number:
                    parseError("Can't have number in module space");
                    break;
                case tok_import:
                    parseImport();
                    continue;
                case tok_identifier:
                    record = parseIdentifier();
                    break;
                case tok_def:
                    record = parseFuncDef();
                    break;
                case tok_return:
                    parseError("keyword return is illegal here.");
                    break;
                default:
                    parseError("Unknown token: "+lx.convertCurrentToken2String());
            }

            if(!record) break;
            module->addASTNode(std::move(record));
            //lgf::streamer sm;
            //module->emitIR(sm);
        }
        if(lx.getCurToken()!= tok_eof){
            std::cout<<"curTok: "<<lx.convertCurrentToken2String()<<std::endl;
            parseError("Module is not closed!");
        }
        return std::move(module);
    }
    std::unique_ptr<astBase> parseReturn(){
        auto ast = std::make_unique<returnAST>(lx.getLoc());
        lx.consume(tok_return);
        std::unique_ptr<astBase> record = nullptr;
        if(lx.getCurToken() == token(';')){
            lx.consume(token(';'));
            return ast;
        } 
        record = parseExpression();
        if(record) ast->addReturnValue(std::move(record));
        lx.consume(token(';'));
        return ast;
    }
    void parseArguments(std::vector<std::unique_ptr<astBase>>& vec){
        while(lx.getCurToken() == tok_identifier){
            auto loc = lx.getLoc();
            auto id = lx.parseIdentifier();
            vec.push_back(parseValAST(loc, id));
            lx.consume(tok_identifier);
            if(lx.getCurToken() != token(',')) break;
            lx.consume(token(','));
        }
        return;
    }
    std::string parseTypeName(){
        auto type = lx.identifierStr;
        lx.consume(tok_identifier);
        if(lx.getCurToken() == token('<')){
            lx.consume(token('<'));
            type+='<';
            std::string tsub = parseTypeName();
            type+= tsub;
            while(lx.getCurToken()== token(',')){
                lx.consume(token(','));
                tsub = parseTypeName();
                type+= tsub;
            }
            type+='>';
            lx.consume(token('>'));
        }
        return type;
    }
    void parseArgSignatures(std::vector<std::unique_ptr<astBase>>& vec){
        while(lx.getCurToken() == tok_identifier){
            auto type = parseTypeName();
            auto id = lx.identifierStr;
            auto loc = lx.getLoc();
            auto ptr = std::make_unique<varDeclAST>(loc, type, id);
            vec.push_back(std::move(ptr));
            lx.consume(tok_identifier);
            if(lx.getCurToken() == token(',')) lx.consume(token(','));
        }
        return;
    }
    std::unique_ptr<astBase> parseFuncDef(){
        lx.consume(tok_def);
        auto id = lx.identifierStr;
        auto loc = lx.getLoc();
        auto ast = std::make_unique<funcDeclAST>(loc, id);
        lx.consume(tok_identifier);
        lx.consume(token('('));
        parseArgSignatures(ast->args);
        lx.consume(token(')'));
        if(lx.getCurToken()==tok_arrow){
            lx.consume(tok_arrow);
            ast->setReturnType(parseTypeName());
        }
        if(lx.getCurToken()==token(';')){
            lx.consume(token(';'));
            return ast;
        }
        // parsing block of function definition
        lx.consume(token('{'));
        ast->isAbstract = 0;
        parseBlock(ast->contents);
        return ast;
    }
    void parseBlock(std::vector<std::unique_ptr<astBase>> &content){
        while(lx.getCurToken()!= token('}')){
            std::unique_ptr<astBase> record = nullptr;
            switch(lx.getCurToken()){
                case token('}'):
                    break;
                case tok_comment:
                    lx.getNextLine();
                    lx.getNextToken();
                    continue;
                case tok_number:
                    record = parseExpression();
                    lx.consume(token(';'));
                    break;
                case tok_identifier:
                    record = parseIdentifier();
                    break;
                case tok_return:
                    record = parseReturn();
                    break;
                case tok_module:
                    parseError("Can not define module here.");
                case tok_def:
                    parseError("Can not define function here.");
                case tok_import:
                    parseError("Can not import module here.");
                default:
                    parseError("Unknown token: "+lx.convertCurrentToken2String());
            }
            if(!record) break;
            content.push_back(std::move(record));
        }
        lx.consume(token('}'));
    }
    std::unique_ptr<astBase> parseNumber(){
        auto nn = lx.number;
        lx.consume(tok_number);
        return std::make_unique<numberAST>(lx.getLoc(), nn);
    }
    std::unique_ptr<astBase> parseParenExpr(){
        lx.consume(token('('));
        auto ast = parseExpression();
        if(lx.getCurToken()!=token(')')) parseError("Missing ')' to complete parenthesis.");
        lx.getNextToken();
        if(!ast) return nullptr;
        return ast;
    }
    std::unique_ptr<astBase> parsePrimary(){
        switch(lx.getCurToken()){
            default:
                parseError("Get unexpected token in a expression.");
                return nullptr;
            case tok_identifier:
                return parseCallOrExpr();
            case tok_number:
                return parseNumber();
            case token('('):
                return parseParenExpr();
        }
    }
    std::unique_ptr<astBase> parseCallOrExpr(){
        auto loc = lx.getLoc();
        auto id = lx.parseIdentifier();
        if(lx.getCurToken()==token('(') ){
            return parseFuncCall(loc, id);
        }else {
            return parseValAST(loc, id);
        }
    }
    std::unique_ptr<astBase> parseValAST(location loc, std::string id){
        bool isModule = 0;
        return std::make_unique<varAST>(loc, id, isModule);
    }
    std::unique_ptr<astBase> parseBinaryRHS(int tokWeight, std::unique_ptr<astBase> lhs){
        while(true){
            bool isReference = 0, isAccess = 0;
            std::string op;
            auto nextTok = binaryTokenPrioCheck();
            op+=static_cast<char>(lx.getCurToken());
            if(nextTok <= tokWeight ) return lhs;
            if(lx.getCurToken() == tok_scope) isReference = 1;
            if(lx.getCurToken() == token('.')) isAccess = 1;
            lx.getNextToken();
            auto loc = lx.getLoc();
            auto rhs = parsePrimary();
            if(!rhs)   
                parseError("Expression isn't complete.");
            auto nnTok = binaryTokenPrioCheck();
            if(nextTok < nnTok ){
                rhs = parseBinaryRHS(nextTok, std::move(rhs));
                if(!rhs){
                    return nullptr;
                } 
            }
            if(isReference){
                lhs = std::make_unique<getReferenceAST>(loc, std::move(lhs), std::move(rhs));
            } else if(isAccess){
                lhs = std::make_unique<accessDataAST>(loc, std::move(lhs), std::move(rhs));
            } else lhs = std::make_unique<binaryAST>(loc, op, std::move(lhs), std::move(rhs));
        }
    }
    std::unique_ptr<astBase> parseExpression(){
        auto lhs = parsePrimary();
        if(!lhs) return nullptr;
        return parseBinaryRHS(0, std::move(lhs));
    }

    int binaryTokenPrioCheck(){
        //if(!isascii(lx.getCurToken())) return -1;
        switch(lx.getCurToken()){
            case token('='):
                return 5;
            case token('-'):
                return 20;
            case token('+'):
                return 20;
            case token('*'):
                return 40;
            case token('/'):
                return 40;
            case token('%'):
                return 30;
            case token('.'):
                return 50;
            case token(tok_scope):
                return 50;
            default:
                return -1;
        }
    }

    std::unique_ptr<astBase> parseDeclaration(){
        return nullptr;
    }
    std::unique_ptr<astBase> parseIdentifier(){
        auto id = lx.parseIdentifier();
        auto loc = lx.getLoc();
        if( lx.getCurToken() == token('(') ){
            lx.consume(token('('));
            auto ast = parseFuncCall(loc, id);
            lx.consume(token(';'));
            return ast;
        }
        // check if the id is declared before
        // if(lx.getCurToken()== tok_identifier){
        //     // TODO: need to first implement the symbolic scan
        //     //if(stbl.find(id)->type_name != "typedef"){
        //     //    parseError(id+" is not a type!");
        //     //}
        //     return parseVarDecl(loc, id);
        // }
        auto lhs = parseValAST(loc, id);
        if(lx.getCurToken() == token(';')){
            lx.getNextToken();
            return lhs;
        }
        auto ast = parseBinaryRHS(0, std::move(lhs));
        lx.consume(token(';'));
        return ast;
    }

    std::unique_ptr<astBase> parseVarDecl(location loc, std::string tid){
        parseError("parseVarDecl Not implement yet!");
        // if(auto ptr = ctx->findSymbolInfoInCurrentModule(tid)){
        //     if(ptr->category!= "type" || ptr->category!= "class"){
        //         parseError("The type \'"+tid+"\' is not defined yet!");
        //     }
        // }
        auto id = lx.identifierStr;
        lx.consume(tok_identifier);
        // TODO: support to parse the case:
        //       var a, b, c
        //       var a = 1, b = 2
        lx.consume(token(';'));
        return std::make_unique<varDeclAST>(loc, tid, id);
    }
    void checkIfVarIDExists(std::string id){
        // if(auto info = ctx->findSymbolInfoInCurrentModule(id)){
        //     if(info->category == "variable") return;
        //     parseError(" \'"+id+"\' is not a variable name!");
        // }
        // parseError("variable name: "+id+" is unknown!");
    }

    std::unique_ptr<astBase> parseFuncCall(location loc, std::string id){
        // auto info = idif.scope->find(idif.id);
        // if(info==nullptr || info->category!="func")
        // parseError("The function: "+idif.id+" is unknown!");
        auto ast = std::make_unique<funcCallAST>(loc, id);  
        lx.consume(token('('));
        while(lx.getCurToken()!=token(')')){
            auto arg = parseExpression();
            ast->addArg(std::move(arg));
            if(lx.getCurToken()!=token(',')) break;
            lx.consume(token(','));
        }
        lx.consume(token(')'));
        //lx.consume(token(';'));
        return ast;
    }
    void parseImport(){
        TRACE_LOG;
        lx.consume(tok_import);
        auto path = lx.buffer;
        if(lx.identifierStr != "lgf"){
            path = lx.identifierStr+"."+path;
        }
        lx.getNextLine();
        lx.getNextToken();
        std::replace(path.begin(), path.end(), '.', '\\');
        path+=".lgf";
        // replace the lgf folder by the root path
        
        std::cout<<"path: "<<path<<std::endl;
        if(!io) parseError("File IO is broken!");
        auto file = io->findInclude(path);
        parser ip(io);
        if(file.empty()) parseError("Can't find the import module: "+file.string());
        lx.getNextToken();
        ip.parseModuleFile(file, program);
    }
    
    void registerID(programAST* program){
        ctx->resetModulePtr();
        for(auto & module : program->modules){
            auto ptr = dynamic_cast<moduleAST*>(module.get());
            ctx->createSubmoduleAndEnter(ptr->name);
            scanBlock(module->contents);
            ctx->moveToParentModule();
        }
    }
    void scanBlock(std::vector<std::unique_ptr<astBase>>& vec){
        for(auto &ast : vec){
            scanAST(ast);
        }
    }
    void scanAST(std::unique_ptr<astBase>& ptr){
        switch(ptr->kind){
            case kind_binary:
                scanBinaryAST(ptr);
                break;
            case kind_funcDecl:
                scanFuncDefAST(ptr);
                break;
            case kind_funcCall:
                scanFuncCall(ptr);
                break;
            case kind_variable:
                scanVarAST(ptr);
                break;
            case kind_getRef:
                scanGetRefAST(ptr);
                break;
            default:
                break;
        }
    }
    
    void scanGetRefAST(std::unique_ptr<astBase>& ptr){
        auto ast = dynamic_cast<getReferenceAST*>(ptr.get());
        auto var = dynamic_cast<varAST*>(ast->module.get());
        var->isModuleID = 1;
        auto path = ast->getPath();
        auto fc = ast->getEndAST();
        auto id = fc->id;
        //check if the func id exists in 
        //the module pointed by path
        if(auto info = ctx->findSymbol(id, path)){
            if(info->category != "func") 
                parseError(id+" is not a function.");
        }else {
            parseError("Can't find '"+id+"' in "+ast->printPath(path));
        }
    }
    void scanVarAST(std::unique_ptr<astBase>& ptr){
        auto ast = dynamic_cast<varAST*>(ptr.get());
        if(ctx->hasSymbol(ast->id)) return;
        ctx->addSymbolInfoToCurrentScope(ast->id,{"var",ast->loc, "variable"});
    }
    void scanBinaryAST(std::unique_ptr<astBase>& ptr){
        auto ast = dynamic_cast<binaryAST*>(ptr.get());
        scanAST(ast->lhs);
        scanAST(ast->lhs);
    }
    void ensureIDAvaliable(std::string id){
        if(auto ptr = ctx->findSymbolInfoInCurrentModule(id))
            parseError("id "+id+" is redefined, original definition is at "+ptr->loc.string());
    }
    idinfo* ensureIDExists(std::string id){
        if(auto ptr = ctx->findSymbolInfoInCurrentModule(id))
            return ptr;
        else parseError("id "+id+" is unknown.");
    }
    void scanFuncDefAST(std::unique_ptr<astBase>& ptr){
        auto ast = dynamic_cast<funcDeclAST*>(ptr.get());
        ensureIDAvaliable(ast->funcID);
        ctx->addSymbolInfoToCurrentScope(ast->funcID, {"func", ast->loc});
    }
    void scanFuncCall(std::unique_ptr<astBase>& ptr){
        auto ast = dynamic_cast<funcCallAST*>(ptr.get());
        auto idif = ensureIDExists(ast->id);
        if(idif->category!="func") parseError("id "+ast->id+" is not a function and is defined at "+idif->loc.string());
        scanBlock(ast->args);
    }

    fileIO *io=nullptr;
    lexer lx;
    ASTContext* ctx;
    programAST* program;
};

}

#endif
