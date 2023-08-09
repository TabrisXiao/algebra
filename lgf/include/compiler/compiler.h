
#ifndef COMPILER_MAIN_H
#define COMPILER_MAIN_H

#include <string>
#include <exception>
#include "compiler/fileIO.h"
#include "compiler/lexer.h"
#include "compiler/parser.h"
#include "compiler/streamer.h"
#include "compiler/LGTranslator.h"
#include "lgf/LGFModule.h"
#include "utils.h"

namespace lgf::compiler{

class compiler {
    public: 
    compiler() : pser(&io) {};
    void compileInput(std::string file){
        ast=std::make_unique<programAST>();
        auto f = io.getFile(file);
        io.addIncludePath(f.parent_path());
        COMPIELR_THROW_WHEN(f.empty(), "Can't find the file: "+file);
        pser.parseMainFile(f, ast.get());
        lgf::streamer sm;
        
        ast->emitIR(sm);
        importModule<lgfModule>();
        builder.build(ast.get());
    }
    template<typename module>
    void importModule(){
        module a;
        a.registerTypes();
    }
    fileIO io;
    parser pser;
    std::unique_ptr<programAST> ast;
    LGTranslator builder;
};

}

#endif