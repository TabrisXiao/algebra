
#include "compiler/lexer.h"
#include "compiler/parser.h"
#include <filesystem>
#include <map>
#include <functional>
#include "compiler/streamer.h"
namespace fs = std::filesystem;

int printUsage(){
    std::cout<<"Compiler usage:\n";
    std::cout<<"-src  : the source to run\n";
    return 0;
}
class ab {
    public :
    ab () = default;
    static int check(){
        return 0;
    }
};
int main(int argc, char* argv[]){
    namespace lgfc = lgf::compiler;
    fs::path inputFile;
    if(argc == 1) return printUsage();
    int count= 1;
    while(count< argc){
        std::string arg = argv[count];
        if(arg == "-src")
            inputFile = fs::absolute(argv[count+1]);
        count+=2;
    }
    lgfc::parser pser;
    pser.typeIdTable.addSymbol("var", nullptr);
    auto module = pser.parse(inputFile.string());
    lgf::streamer sm;
    module->emitIR(sm);

    return 0;
}