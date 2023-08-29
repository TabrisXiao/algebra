
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
        pser.lgfctx = &ctx;
        pser.parseMainFile(f, ast.get());
        lgf::streamer sm;
        
        ast->emitIR(sm);
        importModule<lgfModule>();
        builder.ctx = &ctx;
        builder.build(ast.get());
    }
    template<typename module>
    void importModule(){
        module a;
        a.registerTypes();
    }
    void setRootPath(std::string p){
        io.internalModulePath = p;
    }
    fileIO io;
    parser pser;
    std::unique_ptr<programAST> ast;
    LGFContext ctx;
    LGTranslator builder;
};

}

#endif