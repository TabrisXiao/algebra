
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
#include "context.h"
#include <stack>

namespace lgf::compiler{
#define TRACE_FUNC_CALL \
    std::cout << "Calling function: " << __func__ << std::endl;

class parser{
    public:
    parser(fileIO *io_) : io(io_) {}
    std::unique_ptr<moduleAST> parse(fs::path path){
        lx.loadBuffer(path);
        return parseModule();
    }
    void parseError(const char * msg){
        std::cerr<<lx.getLoc().string()<<": Error: "<<msg;
        std::exit(EXIT_FAILURE);
    }
    void parseError(std::string msg){
        parseError(msg.c_str());
    }

    std::unique_ptr<moduleAST> parseModule(){
        lx.getNextToken();
        current_scopeid = scopeid;
        auto module = std::make_unique<moduleAST>(lx.getLoc(), scopeid++);
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
                    record = parseModule();
                    break;
                case tok_number:
                    record = parseExpression();
                    lx.consume(token(';'));
                    break;
                case tok_import: 
                    parseImport();
                    break;
                case tok_identifier:
                    record = parseIdentifier();
                    break;
                case tok_def:
                    record = parseFuncDef();
                    break;
                case tok_return:
                    record = parseReturn();
                    break;
                default:
                    parseError("Unknown token: "+lx.convertCurrentToken2String());
            }

            if(!record) break;
            module->addASTNode(std::move(record));
            //lgf::streamer sm;
            //module->emitIR(sm);
        }
        if(lx.getCurToken()!= tok_eof) parseError("Module is not closed!");
        current_scopeid = module->previous_id;
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
            auto id = lx.identifierStr;
            auto loc = lx.getLoc();
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
        if(auto info = ctx.current_scope->findSymbolInfo(id)){
            parseError("The identifier: "+id+"is defined in: "+info->loc.string());
        }
        auto loc = lx.getLoc();
        auto ast = std::make_unique<funcDeclAST>(loc, id);
        ctx.current_scope->addSymbolInfo(id, {"func", loc});
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
        ast->contentDefined = 1;
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
    std::unique_ptr<astBase> parseCallOrExpr(){
        auto id = lx.identifierStr;
        auto loc = lx.getLoc();
        lx.consume(tok_identifier);
        if(lx.getCurToken()==token('(')){
            return parseFuncCall(loc, id);
        }else {
            return parseValAST(loc, id);
        }
    }
    std::unique_ptr<astBase> parseValAST(location loc, std::string id){
        if(!ctx.current_scope->hasSymbolInfo(id)) {
            ctx.current_scope->addSymbolInfo(id,{"variable", loc, "variable"});
        }
        return std::make_unique<varAST>(loc, id);
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
    void parseImport(){
        lx.consume(tok_import);
        std::string file = lx.buffer;
        lx.getNextLine();
        lx.getNextToken();
        auto ifile = io->findInclude(file);
        if(ifile.empty()) parseError("Can't find the fiile: "+file);
        lexer lxer;
        lxer.loadBuffer(ifile);
        // TODO: parse the imported module;
    }
    
    std::unique_ptr<astBase> parseDeclaration(){
        return nullptr;
    }
    std::unique_ptr<astBase> parseIdentifier(){
        auto id = lx.identifierStr;
        auto loc = lx.getLoc();
        lx.consume(tok_identifier);
        if(lx.getCurToken()== token('(')){
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
        TRACE_FUNC_CALL;
        if(auto ptr = ctx.current_scope->findSymbolInfo(tid)){
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
        if(auto info = ctx.current_scope->findSymbolInfo(id)){
            if(info->category == "variable") return;
            parseError(" \'"+id+"\' is not a variable name!");
        }
        parseError("variable name: "+id+" is unknown!");
    }

    std::unique_ptr<astBase> parseFuncCall(location loc, std::string id){
        auto ast = std::make_unique<funcCallAST>(loc, id);
        if(!ctx.current_scope->hasSymbolInfo(id))
            parseError("The function: "+id+" is unknown!");
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
    
    fileIO *io=nullptr;
    lexer lx;
    ASTContext ctx;
    symbolTable<std::function<type_t()>> typeIdTable;
    unsigned int current_scopeid, scopeid = 0;
    std::stack<std::string> scope_trace;
};

}

#endif
