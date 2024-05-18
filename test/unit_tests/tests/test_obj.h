#include "unit_test_frame.h"
#include <map>
#include <string>
#include "lgf/value.h"
#include "libs/algebra/algebra.h"
using namespace lgf;

void pr(descriptor d)
{
    std::cout << " pr: " << d.represent() << std::endl;
}

class test_obj : public test_wrapper
{
public:
    test_obj() { test_id = "object test"; };
    bool run()
    {
        auto desc = realNumber::get();
        descriptor desc2(desc);

        std::cout << desc2.represent() << std::endl;

        std::vector<descriptor> vec;
        vec.push_back(desc);
        vec.push_back(desc);
        vec.push_back(desc);
        std::cout << vec[2].represent() << std::endl;

        auto a = desc;
        std::cout << a.represent() << std::endl;
        pr(a);
        return 0;
    }
};