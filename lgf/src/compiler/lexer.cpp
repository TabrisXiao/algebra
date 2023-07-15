
#include "compiler/lexer.h"

char lgfc::lexer::getNextChar(){
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

lgfc::token lgfc::lexer::getToken(){
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
    if(lastChar == token('/')){
        lastChar = getNextChar();
        if(lastChar == token('/')) {
            lastChar = getNextChar();
            return tok_comment;
        }
        return token('/');
    }
    auto tok = token(lastChar);
    lastChar = getNextChar();
    return tok;
}
//---------------------------------------------------

std::string lgfc::lexer::readNextLine(){
    std::string line;
    std::getline(file, line);
    return line;
}
//---------------------------------------------------