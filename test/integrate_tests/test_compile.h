#include "unit_test_frame.h"

#include "libs/Builtin/Builtin.h"
#include "libs/aab/aab.h"
#include "libs/transform/convertToSIO.h"
#include "libs/SIO/exporter.h"

using namespace lgf;
namespace test_body{
class test_compile : public test_wrapper{
    
    public:
    test_compile() {test_id = "pass compile";};
    bool run() {
        LGFContext ctx;
        moduleOp g;
        painter pnt(&ctx);
        pnt.gotoGraph(&g);

        auto varType = ctx.getType<variable>();
        auto vecType = ctx.getType<vectorType>(varType, 4);
        auto x = pnt.paint<declOp>(vecType);
        auto ret = pnt.paint<returnOp>(x->output());
        compile(&ctx, &g);


        return 0;
    }
    void compile(LGFContext* ctx, graph* g){
        passManager pm(ctx, g);
        pm.enablePrintBeforePass();
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
};
};
