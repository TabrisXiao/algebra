#ifndef TEST_FRAME
#define TEST_FRAME
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <windows.h>

#define TEST_CHECK_VALUE(var, val, msg)\
    if (var!= val) { \
        std::cerr<<"\nTest failure: "<<msg<<"\nError details: " __FILE__ ":"<< std::to_string(__LINE__)<< " : Expecting \"" #var "\" to be "<<val<<", but got "<<var<<".\n\n"; \
        isfail = 1;\
    }

class test_wrapper{
    public : 
    test_wrapper() = default;
    virtual bool run() = 0;
    std::string test_id;
    bool isfail = 0;
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
        // Enable ANSI color in windows terminal
        HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD consoleMode;
        GetConsoleMode(hConsole, &consoleMode);
        consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hConsole, consoleMode);
        // Run all tests
        for(auto test: tests){
            try {
                auto start = std::chrono::high_resolution_clock::now();
                std::cout<<"\033[33m[ Run    ]: \033[0m"<<test->test_id<<"...\n";
                bool check = test->run();
                if(check){
                    std::cout<<"\033[31m[ Failed ]: \033[0m";
                }else {
                    std::cout<<"\033[32m[    Pass]: \033[0m";
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration =     std::chrono::duration_cast<std::chrono::microseconds>   (end - start);
                std::cout<<test->test_id<<",  in "<<duration.count()/   1000<<" ms"<<std::endl;
            } catch (const std::exception& e) {
                std::cout << "\033[31m[ error  ]: \033[0m" << "An   exception occurred: " << e.what() << std::endl;
            }
        }
    }
    std::vector<test_wrapper*> tests;
};

#endif