
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
    cmp.pser.typeIdTable.addEntry("var", nullptr);
    cmp.compileInput(inputFile);
    
    auto var_t = cmp.builder.ctx->getType<lgf::variable>();
    auto list_t = cmp.builder.ctx->getType<lgf::listType>(var_t, 10);
    std::cout<<list_t.represent()<<std::endl;

    std::cout<<"\n---- LGIR ----\n";
    cmp.builder.c.printGraph();
    return 0;
}