
#ifndef COMPILER_LEXER_H
#define COMPILER_LEXER_H

#include <memory>
#include <string>
#include <fstream>
#include "lgf/exception.h"
#include <filesystem>
#include "utils.h"

namespace lgf::compiler{

enum token : int {
    tok_eof = -1,
    tok_return = -2,
    tok_var = -3,
    tok_identifier = -6,
    tok_number = -7,
    tok_struct = -100,  // struct
    tok_comment = -101, // //
    tok_import = -102, // import
    tok_module = -104,// module
    tok_scope = -105, // ::
    tok_def  =  -106, // def
    tok_member = -107, // mdef
    tok_arrow = -108, // ->
    tok_decl = -109
};
class lexer {
    public: 
    lexer() = default;
    lexer(std::filesystem::path file) {
        loadBuffer(file);
    }
    void reset(){
        buffer = "";
        curCol =0; curLine = 0;
    }
    ~lexer(){ file.close(); }
    void loadBuffer(std::filesystem::path path){
        if(file.is_open()) file.close();
        std::string pathstr = path.string();
        loc = location({std::make_shared<std::string>(std::move(pathstr)), 0,0});
        file.open(path);
        THROW_WHEN(!file.is_open(), "Can't open the file: "+pathstr);
        buffer="";
    }

    void loadBufferTo(location &loc){
        loadBuffer(std::filesystem::path(*(loc.file.get())));
        for(auto i=0; i<loc.line; i++){
            getNextLine();
        }
        for(auto i=0; i<loc.col; i++){
            getNextChar();
        }
        curCol =loc.col; curLine = loc.line;
    }

    static constexpr bool isalpha(unsigned ch) { return (ch | 32) - 'a' < 26; }

    static constexpr bool isdigit(unsigned ch) { return (ch - '0') < 10; }

    static constexpr bool isalnum(unsigned ch) {
      return isalpha(ch) || isdigit(ch);
    }

    static constexpr bool isgraph(unsigned ch) { return 0x20 < ch && ch < 0x7f; }


    static constexpr bool islower(unsigned ch) { return (ch - 'a') < 26; }

    static constexpr bool isupper(unsigned ch) { return (ch - 'A') < 26; }

    static constexpr bool isspace(unsigned ch) {
      return ch == ' ' || (ch - '\t') < 5;
    }

    char getNextChar();
    void getNextLine(){
        ++curLine;
        curCol = 1;
        buffer = readNextLine();
    }

    std::string readNextLine();

    token getToken();
    token getNextToken(){
        curTok = getToken();
        if (curTok == tok_comment) {
            getNextLine();
            lastChar = ' ';
            return getNextToken();
        }
        return curTok;
    }

    void consume(token tok){
        if( tok != curTok ){
            auto symbol = convertCurrentToken2String();
            std::cerr<<loc.string()<<": Expecting token \'"<<std::string(1, static_cast<char>(tok))<<"\', but get "<<convertCurrentToken2String()<<"\"."<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        getNextToken();
    }
    std::string parseIdentifier(){
        auto ret = identifierStr;
        consume(tok_identifier);
        return ret;
    }
    
    token getCurToken(){
        return curTok;
    }

    std::string convertCurrentToken2String(){
        if(curTok >= 0) return std::string(1, static_cast<char>(curTok));
        else return identifierStr;
    }
    location getLoc(){ return loc; }

    std::ifstream file;
    token curTok;
    // If the current Token is an identifier, this string contains the value.
    std::string identifierStr;
    // If the current Token is a number, this variable contains the value.
    double number;
    char lastChar=' ';
    std::string buffer;
    int curCol=0, curLine=0;
    location loc;
};

} // namespace lgfc
#endif