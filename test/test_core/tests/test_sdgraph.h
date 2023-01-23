#include "sdgraph.h"
#include "test_frame.h"
using namespace sdgl;

class test_sdgl : public test_wrapper{
    public:
    test_sdgl() {test_id = "sdgraph test";};
    bool run() {
        vertex v1, v2, v3,v4;
        v1.linkTo(v2, v3);
        v2.linkTo(v4);
        v3.linkTo(v3);
        auto prit = [&](vertex * v){
            std::cout<<v<<" : "<<&v3<<std::endl;
        };
        v1.BFWalk(prit);
        std::cout<<"----------"<<std::endl;
        v3.detach();
        v1.BFWalk(prit);
        return 0;
    }
};