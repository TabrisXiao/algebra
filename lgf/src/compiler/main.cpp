
#include <iostream>
#include <functional>
#include "compiler/compiler.h"

int printUsage(){
    std::cout<<"Compiler usage:\n";
    std::cout<<"-src  : the source to run\n";
    return 0;
}

int main(int argc, char* argv[]){
    std::string inputFile;
    if(argc == 1) return printUsage();
    int count= 1;
    while(count< argc){
        std::string arg = argv[count];
        if(arg == "-src")
            inputFile = argv[count+1];
        count+=2;
    }
    lgf::compiler::compiler cmp;
    lgf::compiler::option::get().log_lv_trace = 0;
    auto p = fs::current_path()/fs::path("lgf/include/libs");
    cmp.io.addIncludePath(p.string());
    cmp.setRootPath(p.string());
    cmp.compileInput(inputFile);

    return 0;
}