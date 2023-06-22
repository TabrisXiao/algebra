#include "codegen/sketch/sketchParser.h"
using namespace lgf::codegen;

bool sketchParser::searchFile(const std::string& folderPath, const std::string& fileName) {
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (fileExists(entry.path().string())) {
            if (entry.path().filename() == fileName) {
                // File found
                return true;
            }
        } else if (fs::is_directory(entry.path())) {
            // Recursively search subdirectories
            if (searchFile(entry.path().string(), fileName)) {
                // File found in subdirectory
                return true;
            }
        }
    }
    // File not found
    return false;
}

void sketchParser::import(std::string file){
    sketchLexer lx;
    lx.loadBuffer(file);
    lx.getNextToken();
    std::string scope = "";
    while(true){
        switch(lx.getCurToken()){
            case tok_module:
                lx.consume(tok_module);
                scope = getScopeName(lx.buffer);
                scope+="::";
                lx.getNextLine();
                lx.getNextToken();
                break;
            case tok_eof:
                return;
            case tok_op_def:
                lx.consume(tok_op_def);
                tbl.addInfo(scope+lx.identifierStr, info_struct);
                break;
            case tok_type_def:
                lx.consume(tok_type_def);
                tbl.addInfo(scope+lx.identifierStr, info_struct);
                break;
            default:
                lx.getNextToken();
                break;
        }
    }
}

void sketchParser::parseImport(){
    //lexer.consume(tok_import);
    std::string file = lexer.buffer;
    lexer.getNextLine();
    lexer.getNextToken();
    std::string ifile;
    bool isExists = 0;
    for(auto f : includePath){
        auto testpath = (f)+file;
        std::cout<<"checking: "<<testpath<<std::endl;
        if( fileExists(testpath) ){
            isExists = 1;
            ifile = testpath;
            break;
        }
    }
    if(!isExists){
        std::string msg = "Can't find the import file: "+file;
        parseError(msg.c_str());
    }
    auto path = fs::path(file);
    path.replace_extension(".h");
    module.addIncludes(path.string());
    import(ifile);
}

void sketchParser::parserOpDefInputs(opDefAST *op){
    // the example of the inputs:
    // inputs = {
    //     lhs:variable,
    //     rhs:variable
    // }
    lexer.consume(tok_identifier);
    lexer.consume(token('='));
    lexer.consume(token('{'));
    while(lexer.getCurToken()!= token('}')){
        auto vname = lexer.identifierStr;
        lexer.consume(tok_identifier);
        lexer.consume(token(':'));
        auto typeSID = ReadIdentifierWithScope();
        if(! tbl.check(typeSID)){
            std::string msg = "type: "+typeSID+" is unknown.";
            parseError(msg.c_str());
        }
        op->getBuilderOp()->addInput(vname, typeSID);
        if(lexer.getCurToken() != token('}'))
            lexer.consume(token(','));
    }
    lexer.consume(token('}'));
}
void sketchParser::parserOpDefOutputs(opDefAST *op){
    lexer.consume(tok_identifier);
    lexer.consume(token('='));
    lexer.consume(token('{'));
    while(lexer.getCurToken()!= token('}')){
        auto vname = lexer.identifierStr;
        lexer.consume(tok_identifier);
        lexer.consume(token(':'));
        auto typeSID = ReadIdentifierWithScope();
        if(! tbl.check(typeSID)){
            std::string msg = "type: "+typeSID+" is unknown.";
            parseError(msg.c_str());
        }
        op->getBuilderOp()->addOutput(vname, typeSID);
        if(lexer.getCurToken() != token('}'))
            lexer.consume(token(','));
    }
    lexer.consume(token('}'));
}
void sketchParser::parseOpDefDetail(opDefAST *op) {
    module.addOperationHeader();
    lexer.getNextToken();
    while(lexer.getCurToken()!= token('}')){
        if(lexer.identifierStr == "inputs")
            parserOpDefInputs(op);
        else if(lexer.identifierStr == "outputs")
            parserOpDefOutputs(op);
        else return parseError("Unknown token");
    }
}
void sketchParser::parseOpDef() {
    lexer.consume(tok_op_def);
    auto opname = ReadIdentifierWithScope();
    tbl.addInfo(opname, info_struct);
    auto op = pntr.createOp<opDefAST>(opname);
    if(lexer.getCurToken() == token('{'))
        parseOpDefDetail(op);
    lexer.consume(token('}'));
}
void sketchParser::parseCodeBlock(){
    lexer.consume(tok_code);
    std::string spell = "";
    int ntok = 0;
    if(lexer.getCurToken()!=token('{')) parseError("expect '{' but get others");
    char ch = lexer.lastChar;
    while(ntok>=0 && ch!=EOF){
        spell+= ch;
        ch = lexer.getNextChar();
        if(ch == '{') ntok++;
        if(ch == '}') ntok--;
    }
    lexer.getNextToken();
    auto spellOp = pntr.createOp<sketchCodeAST>(spell);
}
void sketchParser::parseTypeDef(){
    module.addOperationHeader();
    lexer.consume(tok_type_def);
    auto tpname = ReadIdentifierWithScope();
    std::cout<<"tpname: "<<tpname<<std::endl;
    tbl.addInfo(tpname, info_struct);
    auto op = pntr.createOp<typeDefAST>(tpname);
    lexer.consume(token('{'));
    lexer.consume(token('}'));
}

void sketchParser::parseClassDef(){
    lexer.consume(tok_identifier);
    return parseOpDef();
}

void sketchParser::parseModule(){
    auto name = lexer.buffer;
    lexer.getNextLine();
    lexer.getNextToken();
    module.setName(name);
}

void sketchParser::parse(){
    lexer.getNextToken();
    while(true){
        switch(lexer.getCurToken()){
            case tok_module:
                parseModule();
                break;
            case tok_code:
                parseCodeBlock();
                break;
            case tok_eof:
                return;
            case tok_op_def:
                parseOpDef();
                break;
            case tok_type_def:
                parseTypeDef();
                break;
            case tok_import:
                parseImport();
                break;
            default:
                std::string msg = "Unknown token: "+lexer.convertCurrentToken2String();
                return parseError(msg.c_str());
        }
    }
}