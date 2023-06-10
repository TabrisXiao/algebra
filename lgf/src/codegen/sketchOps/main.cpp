
#include "codegen/sketch/sketchAST.h"
#include "codegen/sketch/sketchLexer.h"
#include "codegen/sketch/sketchParser.h"
#include "codegen/sketch/sketchWriter.h"
#include <filesystem>

void printUsage(){
    std::cout<<"\nSketch codegen is an executable used to generated the lgf ops by using lgft.\nTo use this executable, just follow the input .lgft file path and followed by the output path."<<std::endl;
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

int main(int argc, char* argv[]){
    std::string inputf, outputpath;
    if(argc != 3){
        printUsage();
        return 1;
    }
    else {
        inputf = std::filesystem::absolute(argv[1]).string();
        outputpath = std::filesystem::absolute(argv[2]).string();
    }
    std::string outputFile = outputpath+getFileName(inputf)+".h";
    
    std::cout<<"Converting inputs from: "<<inputf<<std::endl;

    if(!std::filesystem::exists(outputpath)){
        std::cout<<"Folder not exists, creating the folder: "<<outputpath<<std::endl;
        if(!std::filesystem::create_directories(outputpath)){
            std::cout<<"Failed to create the folder. Abort!"<<std::endl;
            return 1;
        }
    }
    std::cout<<"Generating codes to: "<<outputFile<<std::endl;

    lgf::codegen::sketchParser parser;
    parser.lexer.loadBuffer(inputf);
    parser.parse();
    lgf::codegen::codeWriter writer;
    writer.out.liveStreamToFile(outputFile);
    writer.addTranslationRule<lgf::codegen::sketch2cppTranslationRule>();
    writer.write(&(parser.c));
    return 0;
}