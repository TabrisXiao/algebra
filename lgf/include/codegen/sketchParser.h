
#ifndef CODEGEN_PARSER_H
#define CODEGEN_PARSER_H
#include "lgf/painter.h"
#include "lgf/operation.h"
#include "sketchLexer.h"
#include "sketchAST.h"
#include <vector>
#include <memory>

namespace lgf{
namespace codegen{
class sketchParser {
    public:
    sketchParser() {
        pntr.gotoGraph(&c);
    }
    void parseError(const char * msg){
        //std::cout<<static_cast<char>(lexer.getCurToken())<<std::endl;
        std::cerr<<lexer.getLoc().string()<<" : error: "<<msg;
        std::exit(EXIT_FAILURE);
    }
    void parserOpDefInputs(opDefAST *op){
        // the example of the inputs:
        // inputs = {
        //     lhs:variable,
        //     rhs:variable
        // }
        lexer.consume(tok_op_def_inputs);
        lexer.consume(token('='));
        lexer.consume(token('{'));
        while(lexer.getCurToken()!= token('}')){
            lexer.consume(tok_identifier);
            auto vname = lexer.identifierStr;
            lexer.consume(token(':'));
            lexer.consume(tok_identifier);
            auto typeSID = lexer.identifierStr;
            op->addInput(vname, typeSID);
            if(lexer.getCurToken() != token('}'))
                lexer.consume(token(','));
        }
        lexer.consume(token('}'));
    }
    void parserOpDefOutputs(opDefAST *op){
        lexer.consume(tok_op_def_inputs);
        lexer.consume(token('='));
        lexer.consume(token('{'));
        while(lexer.getCurToken()!= token('}')){
            lexer.consume(tok_identifier);
            auto vname = lexer.identifierStr;
            lexer.consume(token(':'));
            lexer.consume(tok_identifier);
            auto typeSID = lexer.identifierStr;
            op->addOutput(vname, typeSID);
            if(lexer.getCurToken() != token('}'))
                lexer.consume(token(','));
        }
        lexer.consume(token('}'));
    }
    void parseOpDefDetial(opDefAST *op) {
        lexer.getNextToken();
        while(lexer.getCurToken()!= token('}')){
            switch(lexer.getCurToken()){
                case tok_op_def_inputs:
                    parserOpDefInputs(op);
                    break;
                case tok_op_def_outputs:
                    parserOpDefOutputs(op);
                    break;
                default:
                    return parseError("Unknown token");
            }
        }
    }
    void parseOpDef() {
        std::cout<<"parseOpDef"<<std::endl;
        lexer.consume(tok_identifier);
        auto opname = lexer.identifierStr;
        auto op = pntr.createOp<opDefAST>(opname);
        if(lexer.getCurToken() == token('{'))
            parseOpDefDetial(op);
        lexer.consume(token('}'));
    }
    void parseScope(){
        lexer.consume(token('@'));
        lexer.consume(tok_identifier);
        if(lexer.identifierStr == "LGF"){
            lexer.consume(token(':'));
            lexer.consume(token(':'));
            lexer.consume(tok_op_def);
            return parseOpDef();
        }
        THROW_WHEN(true, "Unknown scope name.");
    }
    void parse(){
        lexer.getNextToken();
        while(true){
            switch(lexer.getCurToken()){
                case tok_eof:
                    return;
                case token('@'):
                    parseScope();
                    break;
                default:
                    return parseError("Unknown token.");
            }
        }
    }
    sketchLexer lexer;
    canvas c;
    painter pntr;
};
} // namespace codegen
} // namespace lgf
#endif