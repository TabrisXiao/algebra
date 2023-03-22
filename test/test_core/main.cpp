
#include "tests/test_frame.h"
#include "tests/test_sdgraph.h"
#include "tests/test_aog.h"

int main(){
    //test_aog();
    test_frame frame;
    frame.add_test<test_sdgl>();
    frame.add_test<test_aog>();
    //frame.add_test<test_dgraph>();
    frame.run_all_test();
    return 0;
}