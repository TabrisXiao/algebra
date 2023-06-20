
#include "codegen/sketch/sketchAST.h"
#include "codegen/sketch/sketchLexer.h"
#include "codegen/sketch/sketchParser.h"
#include "codegen/sketch/sketchWriter.h"
#include <filesystem>

namespace fs = std::filesystem;

void printUsage(){
    std::cout<<"\nSketch codegen is an executable used to generated the lgf ops by using lgft.\nThe usage:\n";
    std::cout<<"  -src : the source folder containing the .lgft file to process.\n";
    std::cout<<"  -dst : the destination folder of the outputs.\n";
    std::cout<<"  -i   : the path containing the import sources."<<std::endl;
}

std::string getFileName(const std::string& filePath){
    // Find the position of the last directory separator
    size_t lastSeparatorPos = filePath.find_last_of("/\\");
    if (lastSeparatorPos == std::string::npos) {
        lastSeparatorPos = 0; // No directory separator found, start from the beginning
    } else {
        lastSeparatorPos++; // Move past the directory separator
    }

    // Find the position of the last dot (file format type)
    size_t lastDotPos = filePath.find_last_of(".");
    if (lastDotPos != std::string::npos && lastDotPos > lastSeparatorPos) {
        // Extract the file name without the file format type
        return filePath.substr(lastSeparatorPos, lastDotPos - lastSeparatorPos);
    } else {
        // No file format type found, return the entire file name
        return filePath.substr(lastSeparatorPos);
    }
}

 void getFilesWithExtension(const std::string& folderPath, const std::string& extension, std::vector<std::pair<std::string, std::string>> & files) {
    if (std::filesystem::is_regular_file(folderPath)) {
        auto output = getFileName(folderPath)+".h";
        files.push_back(std::pair<std::string, std::string>(folderPath, output));
    }
    auto folder_szie = folderPath.size();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            auto absPath = entry.path().string();
            auto output = std::filesystem::path(absPath.substr(folder_szie));
            output.replace_extension(".h");
            files.push_back(std::pair<std::string, std::string>(absPath, output.string()));
        }
    }
    return;
}

int main(int argc, char* argv[]){
    std::string includePath, outputpath;
    std::vector<std::pair<std::string, std::string>> inputs;

    if(argc == 1){
        printUsage();
        return 1;
    }

    int count= 1;
    while(count< argc){
        std::string arg = argv[count];
        if(arg == "-src"){
            std::string path=std::filesystem::absolute(argv[count+1]).string();
            getFilesWithExtension(path, ".lgft", inputs);
        }else if(arg == "-dst"){
            outputpath = std::filesystem::absolute(argv[count+1]).string();
        }else if(arg == "-i"){
            includePath = std::filesystem::absolute(argv[count+1]).string();
        } else {
            std::cerr<<"Unknown argument: "<<arg<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        count=count+2;
    }

    if(!std::filesystem::exists(outputpath)){
        std::cout<<"Folder not exists, creating the folder: "<<outputpath<<std::endl;
        if(std::filesystem::create_directories(outputpath)){
            std::cout<<"Failed to create the folder. Abort!"<<std::endl;
            return 1;
        }
    }
    
    for(auto & iter : inputs){
        auto inputf = iter.first;
        std::cout<<"Converting input from: "<<inputf<<std::endl;
        auto outputf = outputpath+iter.second;
        std::cout<<"Generated codes to: "<<outputf<<std::endl;

        lgf::codegen::sketchParser parser;
        parser.lexer.loadBuffer(inputf);
        
        if(!includePath.empty()) parser.addIncludePath(includePath);
        parser.parse();
        lgf::codegen::codeWriter writer;
        writer.out.liveStreamToFile(outputf);
        writer.addTranslationRule<lgf::codegen::sketch2cppTranslationRule>();
        writer.write(&(parser.c));
    }
    return 0;
}