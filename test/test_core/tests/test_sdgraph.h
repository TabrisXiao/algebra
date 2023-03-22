#include "sdgraph.h"
#include "test_frame.h"
using namespace sdgl;

class test_sdgl : public test_wrapper{
    class tvertex : public vertex{
        public :
        tvertex(int num): id(num){}
        int id = 0;
    };
    public:
    test_sdgl() {test_id = "sdgraph test";};
    bool run() {
        vertex v1, v2, v3,v4;
        v1.linkTo(v2, v3);
        v2.linkTo(v4);
        v3.linkTo(v3);
        
        v3.detach();
        bool result = 0;
        v1.BFWalk([&](vertex * v){
            if(v== &v3){
                std::cout<<"ERROR: detach failed!"<<std::endl;
                result = 1;
            }
        });
        return result;
    }
};