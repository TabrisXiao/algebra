
#ifndef CODEGEN_PARSER_H
#define CODEGEN_PARSER_H
#include "lgf/painter.h"
#include "lgf/operation.h"
#include "sketchLexer.h"
#include "sketchAST.h"
#include "symbolTable.h"
#include <vector>
#include <memory>
#include <filesystem>
namespace fs = std::filesystem;

namespace lgf{
namespace codegen{
class sketchParser {
    public:
    sketchParser() {
        pntr.gotoGraph(&module);
    }
    
    bool fileExists(const std::string& filePath) {
        return fs::exists(filePath) && fs::is_regular_file(filePath);
    }

    bool searchFile(const std::string& folderPath, const std::string& fileName);

    void import(std::string file);

    void parseImport();

    void parseError(const char * msg){
        //std::cout<<static_cast<char>(lexer.getCurToken())<<std::endl;
        std::cerr<<lexer.getLoc().string()<<": Error: "<<msg;
        std::exit(EXIT_FAILURE);
    }
    void parseError(std::string msg){
        parseError(msg.c_str());
    }
    void parserOpDefInputs(opDefAST *op);
    void parserOpDefOutputs(opDefAST *op);
    void parseOpDefDetail(opDefAST *op);
    void parseOpDef();

    std::string getScopeName(std::string moduleName){
        // the module name should be: scope.component
        // so this function get the scope part.
        std::string scope = "";
        for(auto & c : moduleName){
            if(c=='.') break;
            scope += c;
        }
        return scope;
    }
    void parseCodeBlock();
    void parseTypeDef();
    void parseClassDef();
    void parseModule();
    std::string parseCPPType();
    void parseTypeInputParameters(typeDefAST* op);
    void parse();
    void parseFile(std::string path){
        lexer.loadBuffer(path);
        fileName = fs::path(path).filename().string();
        parse();
    }

    std::string ReadIdentifierWithScope(){
        auto name = lexer.identifierStr;
        lexer.consume(tok_identifier);
        while(lexer.getCurToken() == tok_scope){
            name+="::";
            lexer.consume(tok_scope);
            name+=lexer.identifierStr;
            lexer.consume(tok_identifier);
        }
        return name;
    }

    void addIncludePath(std::string path){
        includePath.push_back(path);
    }
    std::string fileName, scopeName;
    sketchLexer lexer;
    sketchModuleAST module;
    bool foundModule = 0 ;
    painter pntr;
    symbolTable<sketchTypeInfo> tbl;
    std::vector<std::string> includePath;
};
} // namespace codegen
} // namespace lgf
#endif