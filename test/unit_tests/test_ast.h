#include "unit_test_frame.h"
#include "codegen/sketchLexer.h"
#include "codegen/sketchParser.h"
#include "lgf/painter.h"
#include "codegen/cppWriter.h"
#include <string>

using namespace lgf::codegen;
namespace test_body{
class test_ast : public test_wrapper{

    public:
    test_ast() {test_id = "ast test";};
    bool run() {
        bool res = 0;
        sketchParser parser;
        parser.lexer.loadBuffer("test/unit_tests/resources/test_ast.lgft");
        parser.parse();
        parser.c.print();
        return res;
    }
};
}