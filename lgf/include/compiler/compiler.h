
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

#include "libs/AAB/passes.h"
#include "libs/transform/convertToSIO.h"
#include "libs/SIO/exporter.h"

namespace lgf::compiler{

class compiler {
    public: 
    compiler() {
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
        
        translate(ast, sm);

        compileGraph();

        SIO::export2Txt exporter(&g);
        exporter.run();
        
        std::cout<<"\n--- compilation done!\n";
    }
    void translate(std::unique_ptr<programAST>& ast, lgf::streamer& sm){
        LGTranslator builder(&ctx, &g);
        ast->emitIR(sm);
        builder.ctx = &ctx;
        builder.printTranslatedIR = 1;
        builder.build(ast.get());
    }
    void compileGraph(){
        passManager pm(&ctx, &g);
        pm.enablePrintBeforePass();
        pm.enablePrintAfterPass();
        pm.addNormalizationPass();

        default_pipeline(pm);
        pm.run();
    }
    void default_pipeline(passManager& pm){
        pm.addPass(AAB::createAAProcess());
        pm.addPass(AAB::createCalculusPass());
        pm.addNormalizationPass();
        pm.addPass(transform::createConvertToSIOPass());
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