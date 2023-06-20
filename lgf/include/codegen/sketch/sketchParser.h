
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
        pntr.gotoGraph(&c);
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

    void parseCodeBlock();
    void parseTypeDef();
    void parseClassDef();
    void parseScope();
    void parse();

    void addIncludePath(std::string path){
        includePath.push_back(path);
    }
    sketchLexer lexer;
    canvas c;
    painter pntr;
    symbolTable<sketchTypeInfo> tbl;
    std::vector<std::string> includePath;
};
} // namespace codegen
} // namespace lgf
#endif