#include "unit_test_frame.h"
#include "codegen/codeWriter.h"
#include "codegen/sketchLexer.h"
#include "codegen/sketchParser.h"
#include "codegen/sketchWriter.h"
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
        // parser.c.print();

        codeWriter writer;
        writer.out.liveStreamToConsole();
        writer.addTranslationRule<sketch2cppTranslationRule>();
        writer.write(&(parser.c));

        // int width = 10;
        // int stride = 13;
        // int num_plane = 2;
        // int ps = -100;
        // int len = 45;

        // int plane_high = len, plane_low = 0;

        // int nl= std::ceil(float(len)/width);
        // if(width!=len){
        //     if(stride<0){
        //         int stride_end= (nl-1)*stride;
        //         plane_low = stride_end;
        //         plane_high = width;
        //     }
        //     else {
        //         plane_high = (nl-1)*stride+len-(nl-1)*width;
        //         std::cout<<"plane high: "<<nl<<std::endl;
        //     }
        // }
        // int plane_shift = (num_plane-1)*ps;
        // if(plane_shift>0) plane_high+=plane_shift;
        // else plane_low+=plane_shift;

        // std::cout<<" ["<<plane_low<<", "<<plane_high<<"]"<<std::endl;
        return res;
    }
};
}