#include "unit_test_frame.h"
#include "codegen/ast.h"
#include "lgf/painter.h"
#include "aab/ops.h"
#include <string>

using namespace lgf;
using namespace lgf::ast;
namespace test_body{

class test_codegen : public test_wrapper{
    
    public:
    test_codegen() {test_id = "operation test";};
    bool run() {
        canvas reg;
        painter pntr(&reg);
        auto sumf = pntr.createOp<defFuncOp>("sum", "int", "int", "int");
        pntr.gotoGraph(sumf->getGraph());
        auto sum = pntr.createOp<addOp>(sumf->arg(0), sumf->arg(1));
        auto ret = pntr.createOp<returnOp>(sum->output());
        pntr.gotoParentGraph();
        auto module = pntr.createOp<moduleOp>();
        pntr.gotoGraph(module->getGraph());
        auto x = pntr.createOp<declValue>("int");
        auto y = pntr.createOp<declValue>("int");
        pntr.gotoParentGraph();
        
        reg.assignID();
        reg.print();
        return 0;
    }
};
};
