
#ifndef COMPILER_LEXER_H
#define COMPILER_LEXER_H

#include <memory>
#include <string>
#include <fstream>
#include "lgf/exception.h"
#include <filesystem>

namespace lgf{

namespace compiler{

struct location {
    std::shared_ptr<std::string> file; ///< filename.
    int line;                          ///< line number.
    int col;                           ///< column number.
    std::string string(){
        return (*file)+"("+std::to_string(line)+", "+std::to_string(col)+")";
    }
};
enum token : int {
    tok_eof = -1,
    tok_return = -2,
    tok_var = -3,
    tok_identifier = -6,
    tok_number = -7,
    tok_struct = -100,  // struct
    tok_comment = -101, // //
    tok_module = -104,// module
    tok_scope = -105, // ::
};
class lexer {
    public: 
    lexer() = default;
    lexer(std::string filename) {
        loadBuffer(filename);
    }
    ~lexer(){ file.close(); }
    void loadBuffer(std::string filename){
        auto filepath = std::filesystem::absolute(filename).string();
        loc = location({std::make_shared<std::string>(std::move(filepath)), 0,0});
        file.open(getFilePath());
        THROW_WHEN(!file.is_open(), "Can't open the file: "+getFilePath());
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
        return curTok;
    }

    void consume(token tok){
        if( tok != curTok ){
            auto symbol = convertCurrentToken2String();
            std::cerr<<loc.string()<<": consuming an unexpected token \""<<convertCurrentToken2String()<<"\"."<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        getNextToken();
    }
    
    token getCurToken(){
        return curTok;
    }
    std::string getFilePath(){
        return (*loc.file);
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

}

} // namespace lgf
#endif