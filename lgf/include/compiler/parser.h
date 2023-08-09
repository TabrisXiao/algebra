
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
    using scope_t = scope<idinfo>;
    struct idinfo_t {
        std::string id;
        scope_t* scope;
    };
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
        return program;
    }
    programAST* parseModuleFile(fs::path path, programAST* ast_){
        program = ast_;
        ctx = ast_->getContext();
        lx.loadBuffer(path);
        lx.getNextToken();
        lx.consume(tok_module);
        auto id = lx.parseIdentifier();
        ctx->createScopeAndEnter(id);
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
        while(true){
            std::unique_ptr<astBase> record = nullptr;
            switch(lx.getCurToken()){
                case tok_eof:
                    break;
                case tok_comment:
                    lx.getNextLine();
                    lx.getNextToken();
                    continue;
                //case tok_module:
                //    lx.consume(tok_module);
                //    record = parseModule(lx.parseIdentifier());
                //    break;
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
            auto idif = parseIdInfo();
            vec.push_back(parseValAST(loc, idif));
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
            ctx->addSymbolInfoToCurrentScope(id, {"arg", loc});
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
        if(auto info = ctx->current_scope->findSymbolInfo(id)){
            parseError("The identifier: \'"+id+"\' is defined in: "+info->loc.string());
        }
        auto loc = lx.getLoc();
        auto ast = std::make_unique<funcDeclAST>(loc, id);
        ctx->addSymbolInfoToCurrentScope(id, {"func", loc});
        lx.consume(tok_identifier);
        lx.consume(token('('));
        ctx->createScopeAndEnter(id);
        parseArgSignatures(ast->args);
        lx.consume(token(')'));
        if(lx.getCurToken()==tok_arrow){
            lx.consume(tok_arrow);
            ast->setReturnType(parseTypeName());
        }
        if(lx.getCurToken()==token(';')){
            lx.consume(token(';'));
            ctx->moveToParentScope();
            return ast;
        }
        // parsing block of function definition
        lx.consume(token('{'));
        ast->isAbstract = 0;
        parseBlock(ast->contents);
        ctx->moveToParentScope();
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
                    parseError("Can not define module here");
                case tok_def:
                    parseError("Can not define function here");
                case tok_import:
                    parseError("Can not import module here");
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
    idinfo_t parseIdInfo(scope_t* scope = nullptr ){
        auto id = lx.identifierStr;
        if(!scope) scope = ctx->current_scope;
        lx.consume(tok_identifier);
        if(lx.getCurToken() == tok_scope) 
            scope = &(ctx->root_scope);
        while(lx.getCurToken() == tok_scope){
            if(!scope) parseError("The scope: "+id+" doesn't exists!");
            scope = &(scope->findScope(id));
            lx.getCurToken() == tok_scope;
            id = lx.identifierStr;
            lx.consume(tok_identifier);
        }
        return idinfo_t{id, scope};
    }
    std::unique_ptr<astBase> parseCallOrExpr(){
        auto loc = lx.getLoc();
        auto idif = parseIdInfo();
        if(lx.getCurToken()==token('(') ){
            return parseFuncCall(loc, idif);
        }else {
            return parseValAST(loc, idif);
        }
    }
    std::unique_ptr<astBase> parseValAST(location loc, idinfo_t idif){
        if(!idif.scope->hasSymbolInfo(idif.id)) {
            idif.scope->addSymbolInfo(idif.id,{"variable", loc, "variable"});
        }
        return std::make_unique<varAST>(loc, idif.id);
    }
    std::unique_ptr<astBase> parseBinaryRHS(int tokWeight, std::unique_ptr<astBase> lhs){
        while(true){
            std::string op;
            auto nextTok = binaryTokenPrioCheck();
            op+=static_cast<char>(lx.getCurToken());
            if(nextTok <= tokWeight ) return lhs;
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
            lhs = std::make_unique<binaryAST>(loc, op, std::move(lhs), std::move(rhs));
        }
    }
    std::unique_ptr<astBase> parseExpression(){
        auto lhs = parsePrimary();
        if(!lhs) return nullptr;
        return parseBinaryRHS(0, std::move(lhs));
    }

    int binaryTokenPrioCheck(){
        if(!isascii(lx.getCurToken())) return -1;
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
            default:
                return -1;
        }
    }

    std::unique_ptr<astBase> parseDeclaration(){
        return nullptr;
    }
    std::unique_ptr<astBase> parseIdentifier(){
        auto id = lx.identifierStr;
        auto loc = lx.getLoc();
        lx.consume(tok_identifier);
        idinfo_t idif = {id, ctx->current_scope};
        if(lx.getCurToken() == tok_scope){
            lx.consume(tok_scope);
            auto scope = ctx->root_scope.findScope(id);
            idif = parseIdInfo(&scope);
        }
        auto info = idif.scope->findSymbolInfo(idif.id);
        if(info &&( info->category == "func" || info->category == "mfunc") ){
            lx.consume(token('('));
            auto ast = parseFuncCall(loc, idif);
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
        auto lhs = parseValAST(loc, idif);
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
        if(auto ptr = ctx->current_scope->findSymbolInfo(tid)){
            if(ptr->category!= "type" || ptr->category!= "class"){
                parseError("The type \'"+tid+"\' is not defined yet!");
            }
        }
        auto id = lx.identifierStr;
        lx.consume(tok_identifier);
        // TODO: support to parse the case:
        //       var a, b, c
        //       var a = 1, b = 2
        lx.consume(token(';'));
        return std::make_unique<varDeclAST>(loc, tid, id);
    }
    void checkIfVarIDExists(std::string id){
        if(auto info = ctx->current_scope->findSymbolInfo(id)){
            if(info->category == "variable") return;
            parseError(" \'"+id+"\' is not a variable name!");
        }
        parseError("variable name: "+id+" is unknown!");
    }

    std::unique_ptr<astBase> parseFuncCall(location loc, idinfo_t idif){
        auto info = idif.scope->findSymbolInfo(idif.id);
        if(info==nullptr || info->category!="func")
        parseError("The function: "+idif.id+" is unknown!");
        auto ast = std::make_unique<funcCallAST>(loc, idif.id);  
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
        lx.consume(tok_import);
        auto path = lx.identifierStr+lx.buffer;
        lx.getNextLine();
        lx.getNextToken();
        std::replace(path.begin(), path.end(), '.', '/');
        path+=".lgf";
        if(!io) parseError("File IO is broken!");
        auto file = io->findInclude(path);
        parser ip(io);
        if(file.empty()) parseError("Can't find the import module: "+file.string());
        ip.parseModuleFile(file, program);
    }
    
    fileIO *io=nullptr;
    lexer lx;
    ASTContext* ctx;
    symbolTable<std::function<type_t()>> typeIdTable;
    std::stack<std::string> scope_trace;
    programAST* program;
};

}

#endif
