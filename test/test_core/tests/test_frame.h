#ifndef TEST_FRAME
#define TEST_FRAME
#include <iostream>
#include <string>
#include <vector>

class test_wrapper{
    public : 
    test_wrapper() = default;
    virtual bool run() = 0;
    std::string test_id;
};

class test_frame{
    public: 
    test_frame() = default;
    ~test_frame(){
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
            std::cout<<"[   run  ]: "<<test->test_id<<"...\n";
            bool check = test->run();
            if(check){
                std::cout<<"[ failed ]: ";
            }else {
                std::cout<<"[ success]: ";
            }
            std::cout<<test->test_id<<std::endl;
        }
    }
    std::vector<test_wrapper*> tests;
};

#endif