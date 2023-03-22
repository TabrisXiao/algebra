
#include "test_frame.h"
#include "ops.h"
#include "passes.h"
using namespace aog;
class test_aog : public test_wrapper{
    public:
    test_aog() {test_id = "ops test";}
    bool run(){
        context ctx;
        opBuilder builder(&ctx); 
        // auto mainop = builder.create<moduleOp>();
        // builder.setInsertPoint(mainop->getRegion());
        auto defx = builder.create<defOp>("Symbolic");
        auto defy = builder.create<defOp>();
        auto opxx = builder.create<multiplyOp>(defx->output(), defx->output());
        auto opxy = builder.create<multiplyOp>(defx->output(), defy->output());
        auto opyy = builder.create<multiplyOp>(defy->output(), defy->output());
        auto opyx = builder.create<multiplyOp>(defy->output(), defx->output());
        auto add1 = builder.create<addOp>(opxx->output(), opyy->output());
        auto add2 = builder.create<addOp>(opxy->output(), opyx->output());
        auto addfinal = builder.create<addOp>(add1->output(), add2->output());
        //auto sum = builder.create<sumOp>(opxx->output(), opyy->output(), opxy->output(), opyx->output());
        builder.entranceModule->print(&ctx);
        passManager pm(builder.entranceModule, &ctx);
        pm.enablePrintAfterPass();
        createConvertAddToSumPass(pm);
        createFuseAddToSumPass(pm);
        createLhsAssociatePass(pm);
        createRhsAssociatePass(pm);
        pm.run();
        ctx.resetCounts();
        //builder.entranceModule->print(&ctx);
        return 0;
    }
};