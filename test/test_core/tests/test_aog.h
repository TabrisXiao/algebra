
#include "ops.h"
using namespace aog;
void test_aog(){
    
    context ctx;
    opBuilder builder(&ctx); 
    // auto mainop = builder.create<moduleOp>();
    // builder.setInsertPoint(mainop->getRegion());
    auto defx = builder.create<defOp>("Symbolic");
    auto defy = builder.create<defOp>();
    auto opxx = builder.create<addOp>(defx->output(), defx->output());
    auto opxy = builder.create<multiplyOp>(defx->output(), defy->output());
    builder.entranceModule->print();
}