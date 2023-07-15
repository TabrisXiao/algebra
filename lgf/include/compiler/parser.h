
#ifndef COMPILER_PARSER_H
#define COMPILER_PARSER_H
#include "fileIO.h"
#include "lexer.h"
#include "ast.h"
#include "symbolicTable.h"
#include <memory>
#include <map>
#include <functional>
#include "lgf/operation.h"

namespace lgfc{
using namespace lgf;
#define TRACE_FUNC_CALL \
    std::cout << "Calling function: " << __func__ << std::endl;

struct idInfo {
    enum eType{
        infoType_typeid,
        infoType_variable,
        infoType_func,
    };
    eType type;
    int scopeid;
    int rank=0;
};

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
                    break;
                case tok_import: 
                    parseImport();
                    break;
                case tok_identifier:
                    record = parseIdentifier();
                    break;
                default:
                    parseError("Unknown token: "+lx.convertCurrentToken2String());
            }
            if(!record) break;
            module->addASTNode(std::move(record));
        }
        if(lx.getCurToken()!= tok_eof) parseError("Module is not closed!");
        current_scopeid = module->previous_id;
        return std::move(module);
    }
    std::unique_ptr<astBase> parseNumber(){
        auto nn = lx.number;
        lx.consume(tok_number);
        return std::make_unique<numberAST>(lx.getLoc(), nn);
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
        if(!idtbl.check(id)) {
            idInfo info{idInfo::infoType_variable, current_scopeid};
            idtbl.addSymbol(id,info);
        }
        return std::make_unique<varAST>(loc, id);
    }
    std::unique_ptr<astBase> parseBinaryRHS(int tokWeight, std::unique_ptr<astBase> lhs){
        while(true){
            std::string op;
            auto nextTok = binaryTokenPrioCheck();
            op+=static_cast<char>(lx.getCurToken());
            if(nextTok < tokWeight ) return lhs;
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
            if(nnTok == -1) {
                lx.consume(token(';'));
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
        // lx.consume(tok_import);
        // std::string file = lx.buffer;
        // lx.getNextLine();
        // lx.getNextToken();
        // bool isExists = 0;
        // for(auto f : includePath){
        //     auto testpath = (f)+file;
        //     if( fileExists(testpath) ){
        //         isExists = 1;
        //         ifile = testpath;
        //         break;
        //     }
        // }
        // if(!isExists){
        // std::string msg = "Can't find the import file: "+file;
        // parseError(msg.c_str());
        // }
        // auto path = fs::path(file);
        // path.replace_extension(".h");
        // module.addIncludes(path.string());
        // import(ifile);
    }
    std::unique_ptr<astBase> parseDeclaration(){
        return nullptr;
    }
    std::unique_ptr<astBase> parseIdentifier(){
        auto id = lx.identifierStr;
        auto loc = lx.getLoc();
        lx.consume(tok_identifier);
        if(lx.getCurToken()== token('(')){
            if(!idtbl.check(id)) 
                parseError("Function: "+id+" is undefined!");
            if(idtbl.get(id)->type!= idInfo::infoType_func)
                parseError(id+" is not a function!");
            return parseFuncCall(loc, id);
        }
        // check if the id is declared before
        if(lx.getCurToken()== tok_identifier){
            if(!typeIdTable.check(id)){
                parseError(id+" is not a type!");
            }
            return parseVarDecl(loc, id);
        }
        auto lhs = parseValAST(loc, id);
        return parseBinaryRHS(0, std::move(lhs));
    }

    std::unique_ptr<astBase> parseVarDecl(location loc, std::string tid){
        auto id = lx.identifierStr;
        lx.consume(tok_identifier);
        idInfo info{idInfo::infoType_variable, current_scopeid, 1};
        idtbl.addSymbol(id, info);
        // todo  parse the case:
        //       var a, b, c
        //       var a = 1, b = 2
        lx.consume(token(';'));
        return std::make_unique<varDeclAST>(loc, tid, id);
    }

    std::unique_ptr<astBase> parseFuncCall(location loc, std::string id){
        parseError("Not implement");
        return nullptr;
    }
    
    fileIO *io=nullptr;
    lexer lx;
    symbolTable<idInfo> idtbl;
    symbolTable<std::function<type_t()>> typeIdTable;
    unsigned int current_scopeid, scopeid = 0;
};

}

#endif
