
#ifndef COMPILER_MAIN_H
#define COMPILER_MAIN_H

#include <string>
#include <exception>
#include "compiler/fileIO.h"
#include "compiler/lexer.h"
#include "compiler/parser.h"
#include "compiler/streamer.h"
#include "compiler/LGTranslator.h"
#include "libs/moduleManager.h"
#include "utils.h"

namespace lgf::compiler{

class compiler {
    public: 
    compiler() {
        moduleManager::get().loadInternalModule("Builtin", &ctx, nullptr);
    };
    void compileInput(std::string file){
        std::unique_ptr<programAST> ast=std::make_unique<programAST>();
        auto f = io.getFile(file);
        io.addIncludePath(f.parent_path());
        COMPIELR_THROW_WHEN(f.empty(), "Can't find the file: "+file);
        parser pser(&io);
        pser.lgfctx = &ctx;
        pser.parseMainFile(f, ast.get());
        lgf::streamer sm;
        LGTranslator builder(&ctx, &g);
        ast->emitIR(sm);
        builder.ctx = &ctx;
        builder.printTranslatedIR = 1;
        builder.build(ast.get());

        compileGraph();
    }
    void compileGraph(){
        passManager pm(&ctx, &g);
        pm.enablePrintAfterPass();
        pm.addNormalizationPass();
        pm.run();
    }
    void setRootPath(std::string p){
        io.internalModulePath = p;
    }

    fileIO io;
    LGFContext ctx;
    canvas g;
};

}

#endif