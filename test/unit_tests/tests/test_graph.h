
#include "unit_test_frame.h"
#include "lgf/node.h"

using namespace lgf;
namespace test_body{
// this tests targets to tests painter class on various graphs
class test_painter : public test_wrapper{
    public:
    test_painter() {test_id = "painter test";};
    bool run() {
        return 0;
    }
};
} // namespace test_body