
#include "codegen/sketch/sketchLexer.h"

char lgf::codegen::sketchLexer::getNextChar(){
    if (buffer.empty()){
        getNextLine();
    }
    if(buffer.empty()) return EOF;
    ++curCol;
    auto nextchar = buffer.front();
    buffer = buffer.erase(0, 1);
    if (nextchar == '\n') {
        ++curLine;
        curCol = 0;
    }
    return nextchar;
}
//---------------------------------------------------

lgf::codegen::token lgf::codegen::sketchLexer::getNextL0Token(){
    loc.col = curCol;
    loc.line = curLine;
    // skip all space
    while (isspace(lastChar))
        lastChar = getNextChar();
    // Number: [0-9] ([0-9.])*
    if (isdigit(lastChar)) {
        std::string numStr;
        do {
            numStr += lastChar;
            lastChar = getNextChar();
        } while (isdigit(lastChar) || lastChar == '.');
        number = strtod(numStr.c_str(), nullptr);
        return tok_number;
    }
    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar)) {
        identifierStr = lastChar;
        while (isalnum((lastChar = l0token(getNextChar()))) || lastChar == '_')
            identifierStr += lastChar;
        return tok_identifier;
    }
    if (lastChar == EOF ){
        if(file.eof()) return tok_eof;
        else {
            lastChar = getNextChar();
            return getNextL0Token();
        }
    }
    if(lastChar == token(':')){
        lastChar = getNextChar();
        if(lastChar == token(':')) {
            lastChar = getNextChar();
            return tok_scope;
        }
        return token(':');
    }
    auto tok = l0token(lastChar);
    lastChar = getNextChar();
    return tok;
}
//---------------------------------------------------

std::string lgf::codegen::sketchLexer::readNextLine(){
    std::string line;
    std::getline(file, line);
    return line;
}
//---------------------------------------------------

lgf::codegen::token lgf::codegen::sketchLexer::getNextL1Token(){
    auto l0tok = getNextL0Token();
    if(l0tok == tok_identifier){
        if(identifierStr == "module") return tok_module;
        if(identifierStr == "import") return tok_import;
        if(identifierStr == "code") return tok_code;
        if(identifierStr == "type") return tok_type_def;
        if(identifierStr == "operation" ) return tok_op_def;
        if(identifierStr == "return") return tok_return;
        if(identifierStr == "type") return tok_type_def;
        return tok_identifier;
    }
    
    return l0tok;
}
//---------------------------------------------------