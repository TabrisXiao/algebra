#include <iostream>
#include "unit_test_frame.h"
#include "test_dgraph.h"

int main(){
    unit_test_frame frame;
    frame.add_test<test_dgraph>();
    frame.run_all_test();
    return 0;
}