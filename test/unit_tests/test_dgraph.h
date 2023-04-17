
#include "dgraph.h"
#include "unit_test_frame.h"
using namespace dgl;
class tvertex : public vertex{
    public :
    tvertex(int num): id(num){}
    int id = 0;
};
namespace test_body{

class test_dgraph : public test_wrapper{
    public:
    test_dgraph() {test_id = "dgraph test";};
    bool run() {
        tvertex v1(1), v2(2), v3(3),v4(4), v5(5);
        // test configure, v5 is isolated
        //      v1      v5
        //     /  \
        //    v2-> v3
        //   /
        //  v4
        dedge e1, e2, e3, e4;
        e1.connect(&v1, &v2);
        e2.connect(&v1, &v3);
        e3.connect(&v2, &v3);
        e4.connect(&v2, &v4);
        
        graph g;
        g.addL1Vertex(&v1);
        g.addL1Vertex(&v5);
        int counts = 0;
        int order[5] = {1, 5, 2, 3, 4};
        bool test_result = 0;
        g.BFWalk([&](vertex * vtx){
            auto v = dynamic_cast<tvertex*>(vtx);
            TEST_CHECK_VALUE(v->id, order[counts], test_result,"edge::connect graph connected incorrectly!");
            counts++;
        });

        int order2[4] = {1, 5, 2, 4};
        e2.dropConnectionTo(&v3);
        e3.dropConnectionTo(&v3);
        TEST_CHECK_VALUE(v3.inEdges.size(), 0, test_result, "dedge::dropConnectionTo failes!");
        // detach turn to configure, 
        // detach do not affect the isolated level 1 vertex in graph
        //      v1   v5
        //     /  
        //    v2
        //   /
        //  v4
        counts=0;
        g.BFWalk([&](vertex * vtx){
            auto v = dynamic_cast<tvertex*>(vtx);
            TEST_CHECK_VALUE(v->id, order2[counts], test_result, "vertex::detach graph detached incorrectly!");
            counts++;
        });

        return test_result;
    }
};
}