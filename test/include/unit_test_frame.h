#ifndef TEST_FRAME
#define TEST_FRAME
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#define TEST_CHECK_VALUE(var, val, ret, msg)\
    if (var!= val) { \
        std::cerr<<"\nTest failure: "<<msg<<"\nError details: " __FILE__ ":"<< std::to_string(__LINE__)<< " : Expecting \"" #var "\" to be "<<val<<", but got "<<var<<".\n\n"; \
        ret = 1;\
    }

class test_wrapper{
    public : 
    test_wrapper() = default;
    virtual bool run() = 0;
    std::string test_id;
};

class unit_test_frame{
    public: 
    unit_test_frame() = default;
    ~unit_test_frame(){
        for(auto v : tests){
            delete v;
        }
    }
    template<typename T, typename ...ARGS>
    void add_test(ARGS& ...arg){
        auto test = new T(arg...);
        tests.push_back(test);
    }
    void run_all_test(){ 
        for(auto test: tests){
            auto start = std::chrono::high_resolution_clock::now();
            std::cout<<"[ Run    ]: "<<test->test_id<<"...\n";
            bool check = test->run();
            if(check){
                std::cout<<"[ Failed ]: ";
            }else {
                std::cout<<"[    Pass]: ";
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout<<test->test_id<<",  in "<<duration.count()/1000<<" ms"<<std::endl;
        }
    }
    std::vector<test_wrapper*> tests;
};

#endif