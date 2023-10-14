#include <iostream>
#include "unit_test_frame.h"
#include "test_graph.h"
//#include "test_codegen.h"

int main(){
    unit_test_frame frame;
    //frame.add_test<test_operation>();
    //frame.add_test<test_codegen>();
    frame.add_test<test_body::test_painter>();
    frame.run_all_test();
    return 0;
}