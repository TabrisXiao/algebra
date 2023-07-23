
#ifndef COMPILER_MAIN_H
#define COMPILER_MAIN_H

#include <string>
#include <exception>
#include "compiler/fileIO.h"
#include "compiler/lexer.h"
#include "compiler/parser.h"
#include "compiler/streamer.h"
#include "compiler/LGTranslator.h"
#include "utils.h"

namespace lgf::compiler{

class compiler {
    public: 
    compiler() : pser(&io) {};
    void compileInput(std::string file){
        auto f = io.getFile(file);
        COMPIELR_THROW_WHEN(f.empty(), "Can't find the file: "+file);
        main = pser.parse(f);
        builder.astctx = &(pser.ctx);
        builder.build(main);
        auto var_t = builder.ctx->getType<variable>();
        auto list_t = builder.ctx->getType<listType>(var_t, 10);
        std::cout<<list_t.represent()<<std::endl;
    }
    fileIO io;
    parser pser;
    std::unique_ptr<moduleAST> main;
    LGTranslator builder;
};

}

#endif