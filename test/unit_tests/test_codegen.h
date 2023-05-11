#include "unit_test_frame.h"
#include "codegen/ast.h"
#include "lgf/painter.h"
#include "aab/ops.h"
#include "codegen/codeWriter.h"
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
        auto sumf = pntr.createOp<defFuncAST>("sum", "int", "int", "int");
        pntr.gotoGraph(sumf->getGraph());
        auto sum = pntr.createOp<addOp>(sumf->arg(0), sumf->arg(1));
        auto ret = pntr.createOp<returnAST>(sum->output());
        pntr.gotoParentGraph();
        auto module = pntr.createOp<moduleOp>();
        pntr.gotoGraph(module->getGraph());
        auto x = pntr.createOp<declValueAST>("int");
        auto y = pntr.createOp<declValueAST>("int");
        pntr.gotoParentGraph();
        
        reg.assignID();
        reg.print();
        std::cout<<"----------------codegen-----------------"<<std::endl;
        codegen::codeWriter writer;
        writer.out.liveStreamToConsole();
        writer.addTranslationRule<codegen::cpp::cppTranslationRule>();
        writer.write(&reg);
        return 0;
    }
};
};
