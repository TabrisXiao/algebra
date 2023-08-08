
#include "compiler/lexer.h"

char lgf::compiler::lexer::getNextChar(){
    if (buffer.empty()){
        getNextLine();
    }
    if(buffer.empty()) return EOF;
    ++curCol;
    auto nextchar = buffer.front();
    buffer = buffer.erase(0, 1);
    if (nextchar == '\n') {
        ++curLine;
        curCol = 1;
    }
    return nextchar;
}
//---------------------------------------------------

lgf::compiler::token lgf::compiler::lexer::getToken(){
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
        while (isalnum((lastChar = token(getNextChar()))) || lastChar == '_')
            identifierStr += lastChar;

        if(identifierStr == "module") return tok_module;
        if(identifierStr == "class") return tok_struct;
        if(identifierStr == "def") return tok_def;
        if(identifierStr == "mdef") return tok_member;
        if(identifierStr == "import") return tok_import;
        if(identifierStr == "return") return tok_return;

        return tok_identifier;
    }
    if (lastChar == EOF ){
        if(file.eof()) return tok_eof;
        else {
            lastChar = getNextChar();
            return getToken();
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
    if(lastChar == '/'){
        lastChar = getNextChar();
        if(lastChar == '/') {
            lastChar = getNextChar();
            return tok_comment;
        }
        return token('/');
    }
    auto tok = token(lastChar);
    lastChar = getNextChar();

    // checking arrow ->
    if(tok == token('-') && lastChar == '>') {
        tok = tok_arrow;
        lastChar = getNextChar();
    }
    return tok;
}
//---------------------------------------------------

std::string lgf::compiler::lexer::readNextLine(){
    std::string line;
    std::getline(file, line);
    return line;
}
//---------------------------------------------------