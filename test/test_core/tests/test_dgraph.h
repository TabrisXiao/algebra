#include "dgraph.h"
#include "test_frame.h"
using namespace dgl;
class tvertex : public vertex{
    public :
    tvertex(int num): id(num){}
    int id = 0;
};

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
        edge e1, e2, e3, e4;
        e1.connect(v1, v2);
        e2.connect(v1, v3);
        e3.connect(v2, v3);
        e4.connect(v2, v4);
        
        graph g;
        g.addL1Vertex(&v1);
        g.addL1Vertex(&v5);
        int counts = 0;
        int order[5] = {1, 5, 2, 3, 4};
        bool test_result = 0;
        g.BFWalk([&](vertex * vtx){
            auto v = dynamic_cast<tvertex*>(vtx);
            if(order[counts] != v->id) {
                std::cout<<"order1 error, expect: "<<order[counts]<<" got: "<<v->id<<std::endl;
                test_result = 1;
            }
            counts++;
        });

        int order2[4] = {1, 5, 2, 4};
        v3.detach();
        v5.detach();
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
            if(order2[counts] != v->id) {
                std::cout<<"order2 error, expect: "<<order2[counts]<<" got: "<<v->id<<std::endl;
                test_result = 1;
            }
            counts++;
        });

        tvertex v12(12);
        int order3[4] = {1, 5, 12, 4};
        g.replaceVertex(&v2, &v12);
        counts =0;
        g.BFWalk([&](vertex * vtx){
            auto v = dynamic_cast<tvertex*>(vtx);
            if(order3[counts] != v->id) {
                std::cout<<"order3 error, expect: "<<order2[counts]<<" got: "<<v->id<<std::endl;
                test_result = 1;
            }
            counts++;
        });
        return test_result;
    }
};