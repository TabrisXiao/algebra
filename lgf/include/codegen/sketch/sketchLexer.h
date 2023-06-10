
#ifndef CODEGEN_LEXER_H
#define CODEGEN_LEXER_H

#include <memory>
#include <string>
#include <fstream>
#include "lgf/exception.h"
#include <filesystem>

namespace lgf{

namespace codegen{
typedef int64_t token;
struct location {
    std::shared_ptr<std::string> file; ///< filename.
    int line;                          ///< line number.
    int col;                           ///< column number.
    std::string string(){
        return (*file)+"("+std::to_string(line)+", "+std::to_string(col)+")";
    }
};
enum l0token : int64_t {
    tok_eof = -1,
    tok_return = -2,
    tok_var = -3,
    tok_identifier = -6,
    tok_number = -7,
    tok_op_def = -102,
    tok_op_def_inputs = -103,
    tok_op_def_outputs= -104,
    tok_lgf_scope = -105,
};
class sketchLexer {
    public: 
    sketchLexer() = default;
    sketchLexer(std::string filename) {
        loadBuffer(filename);
    }
    ~sketchLexer(){ file.close(); }
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

    std::string readNextLine();

    token getNextL0Token();
    
    token getNextL1Token();

    token getNextToken(){
        curTok = getNextL1Token();
        return curTok;
    }
    void consume(token tok){
        if( tok != curTok ){
            std::string symbol= "special";
            if(tok > 0 ) {
                symbol = char(tok);
            }
            else if(tok == tok_identifier)
                symbol = "symbol or literal";
            std::cerr<<loc.string()<<": expect "<<symbol<<" but get others"<<std::endl;
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

} // namespace codegen
} // namespace lgf
#endif