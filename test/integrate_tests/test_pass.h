#include "unit_test_frame.h"
#include "lgf/operation.h"
#include "aab/ops.h"
#include "lgf/painter.h"
#include "lgf/lgOps.h"
#include "lgf/pass.h"
#include "aab/passes.h"

using namespace lgf;
namespace test_body{
class test_pass : public test_wrapper{
    
    public:
    test_pass() {test_id = "pass test";};
    bool run() {
        painter builder;
        auto module = builder.createOp<moduleOp>();
        builder.gotoGraph(module->getGraph());
        auto varx = builder.createOp<defOp>("x");
        auto vary = builder.createOp<defOp>("y");
        auto xy = builder.createOp<multiplyOp>(varx->outputValue(), vary->outputValue());
        auto yx = builder.createOp<multiplyOp>(vary->outputValue(), varx->outputValue());
        auto xx = builder.createOp<multiplyOp>(varx->outputValue(), varx->outputValue());
        auto yy = builder.createOp<multiplyOp>(vary->outputValue(), vary->outputValue());
        auto add1 = builder.createOp<addOp>(xx->outputValue(), yy->outputValue());
        auto add2 = builder.createOp<addOp>(yx->outputValue(), xy->outputValue());
        auto add3 = builder.createOp<addOp>(add1->outputValue(), add2->outputValue());
        module->assignID(0);
        builder.getParentGraph()->print();
        //module->print();
        passManager pm(module);
        pm.enablePrintAfterPass();
        createConvertAddToSumPass(pm);
        createFuseSumOpPassPass(pm);
        createLhsAssociatePass(pm);
        pm.run();
        return 0;
    }
};
};
