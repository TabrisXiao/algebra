
#pragma once

#include "unit_test_frame.h"
#include "libs/Builtin/Builtin.h"
#include "libs/fa/passes.h"
#include "libs/compiler.h"
#include "libs/SIO/exporter.h"

using namespace lgf;

namespace test_body{

class test_fa : public test_wrapper{
    public:
    test_fa() {test_id = "fa functional test";};
    bool test1(){
        // test chain rule
        LGFContext ctx;
        moduleOp g;
        g.name = "main"; 
        g.setContext(&ctx);
        painter pnt(&ctx);
        pnt.gotoGraph(&g);
        auto real = ctx.getType<real_t>();
        auto x = pnt.paint<declOp>(real);
        auto y = pnt.paint<declOp>(real);
        auto sum = pnt.paint<productOp>(x->output(), y->output());
        auto sin = pnt.paint<funcSineOp>(sum->output());
        auto cos = pnt.paint<funcCosOp>(sin->output());
        auto dcos = pnt.paint<differentiateOp>(cos->output(), x->output());
        auto ret = pnt.paint<returnOp>(dcos->output());
        AABCompiler core;
        //core.getManager()->enablePrintAfterPass();
        g.print();
        core.compile(&ctx, &g);
        g.print();
        SIO::export2Txt exporter(&g);
        exporter.run();
        return 0;
    }
    bool test2(){
        // test chain rule
        LGFContext ctx;
        moduleOp g;
        g.name = "main"; 
        g.setContext(&ctx);
        painter pnt(&ctx);
        pnt.gotoGraph(&g);
        auto real = ctx.getType<real_t>();
        auto x = pnt.paint<declOp>(real);
        auto sin = pnt.paint<funcSineOp>(x->output());
        auto cos = pnt.paint<funcCosOp>(x->output());
        auto prod = pnt.paint<productOp>(x->output(), sin->output(), cos->output());
        auto dprod = pnt.paint<differentiateOp>(prod->output(), x->output());
        auto ret = pnt.paint<returnOp>(dprod->output());
        AABCompiler core;
        //core.getManager()->enablePrintAfterPass();
        g.print();
        core.compile(&ctx, &g);
        g.print();
        SIO::export2latex exporter(&g);
        exporter.run();
        return 0;
    }
    bool run(){
        bool res = test1();
        res |= test2();
        return res;
    }
    
};
} // namespace test_qft