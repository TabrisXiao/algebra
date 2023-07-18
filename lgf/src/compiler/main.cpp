
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
    lgf::streamer sm;
    cmp.main->emitIR(sm);

    std::cout<<"\n---- LGIR ----\n";
    cmp.builder.c.printGraph();
    return 0;
}