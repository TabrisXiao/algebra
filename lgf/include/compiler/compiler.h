
#ifndef COMPILER_MAIN_H
#define COMPILER_MAIN_H

#include <string>
#include <exception>
#include "compiler/fileIO.h"
#include "compiler/lexer.h"
#include "compiler/parser.h"
#include "compiler/streamer.h"

#define COMPIELR_THROW(msg)\
    std::cerr<<"Compiler Error: " __FILE__ ":"<< std::to_string(__LINE__)<<msg<<std::endl;\
    std::exit(EXIT_FAILURE);

#define COMPIELR_THROW_WHEN(condition, msg)\
    if (condition){\
        std::cerr<<"Compiler Error: " __FILE__ ":"<< std::to_string(__LINE__)<<msg<<std::endl;\
        std::exit(EXIT_FAILURE);\
    }
    
namespace lgfc{

class compiler {
    public: 
    compiler() : pser(&io) {};
    void compileInput(std::string file){
        auto f = io.getFile(file);
        COMPIELR_THROW_WHEN(f.empty(), "Can't find the file: "+file);
        main = pser.parse(f);
    }
    fileIO io;
    parser pser;
    std::unique_ptr<moduleAST> main;
};

}

#endif