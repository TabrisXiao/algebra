
#ifndef LGF_LITEPARSER_H
#define LGF_LITEPARSER_H
#include <string>
#include <exception>
#include <iostream>
#include "utils.h"
namespace lgf{

class liteParser {
    public:
    enum token : int {
        tok_identifier=-3,
        tok_number=-2,
        tok_eof=-1,
    };
    liteParser() = default;
    liteParser(std::string c){ loadBuffer(c); }
    void loadBuffer(std::string ct) { buffer = ct; getNextToken();}
    bool isEOF() { return curTok == tok_eof; }
    std::string getBuffer(){
        return buffer.substr(charPtr);
    }
    char getNextChar(){
        if (charPtr > (buffer.size()-1) ) return EOF;
        return buffer[charPtr++];
    }
    int getToken(){
        while(utils::isspace(lastChar)) lastChar = getNextChar();
        if(isdigit(lastChar)) {
            std::string numStr;
            do {
                numStr += lastChar;
                lastChar = getNextChar();
            } while (isdigit(lastChar) || lastChar == '.');
            number = strtod(numStr.c_str(), nullptr);
            return tok_number;
        }
        if(isalpha(lastChar)) {
            identifierStr = lastChar;
            while (isalnum((lastChar = token(getNextChar()))) || lastChar == '_')
            identifierStr += lastChar;
            return tok_identifier;
        }
        if(lastChar == EOF) return tok_eof;
        auto tok = int(lastChar);
        lastChar = getNextChar();
        return tok;
    }
    void parseError(std::string msg, std::string caller){
        std::exit(EXIT_FAILURE);
    }
    int getCurToken(){
        return curTok;
    }
    int getNextToken(){
        curTok = getToken();
        return curTok;
    }
    std::string convertToken2String(int tok){
        if(tok >= 0) return std::string(1, static_cast<char>(curTok));
        else if(tok == -3) return "literals";
        else if(tok == -2) return "number";
    }
    void consume(int tok){
        if( tok != curTok ){
            std::cerr<<": expecting "<<convertToken2String(tok)<<" but got "<<convertToken2String(curTok)<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        getNextToken();
    }
    std::string parseIdentifier(){
        auto str = identifierStr;
        consume(tok_identifier);
        return str;
    }
    double parseNumber(){
        double num = number;
        consume(tok_identifier);
        return num;
    }
    void parseColon(){ consume(int(':')); }
    void parseLeftSqaureBracket() { consume(int('['));}
    void parseRightSqaureBracket() { consume(int(']'));}
    void parseLessThan(){ consume(int('<'));}
    void parseGreaterThan(){ consume(int('>'));}
    void parseLeftParenthesis() {consume(int('('));}
    void parseRightParenthesis() {consume(int(')'));}
    void parseComma() {consume(int(','));}
    void parseDot(){ consume(int('.')); }
    char lastChar=' ';
    int charPtr=0, curTok;
    std::string buffer, identifierStr;
    double number;
};
}

#endif