#include <iostream>
#include "unit_test_frame.h"
#include "test_dgraph.h"
#include "test_operation.h"

using namespace test_body;
int main(){
    unit_test_frame frame;
    frame.add_test<test_dgraph>();
    frame.add_test<test_operation>();
    frame.run_all_test();
    return 0;
}