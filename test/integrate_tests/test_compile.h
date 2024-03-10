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
        g.setContext(&ctx);
        painter pnt(&ctx);
        pnt.gotoGraph(&g);

        auto varType = ctx.getType<variable>();
        auto vecType = ctx.getType<vectorType>(varType, 4);
        
        auto x = pnt.paint<declOp>(vecType);
        auto y = pnt.paint<declOp>(vecType);
        auto z = pnt.paint<declOp>(vecType);
        auto a = pnt.paint<AAB::directProductOp>(x->output(), y->output());
        auto res = pnt.paint<AAB::directProductOp>(a->output(), z->output());
        //auto ret = pnt.paint<returnOp>(res->output());

        auto ttype = ctx.getType<tensorType>(varType, 4, std::vector<bool>{0});
        auto gtype = ctx.getType<tensorType>(varType, 4, std::vector<bool>{0, 1});
        auto gg = pnt.paint<declOp>(gtype);
        auto v = pnt.paint<declOp>(ttype);
        auto m = pnt.paint<AAB::contractionOp>(gg->output(), v->output(), 1, 0);
        auto ret = pnt.paint<returnOp>(m->output());
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
        pm.addNormalizationPass();
        pm.addPass(transform::createConvertToSIOPass());
    }
};
}// namespace test_body
