
#ifndef COMPILER_IO_H
#define COMPILER_IO_H

#include <string>
#include <filesystem>

namespace fs = std::filesystem;
namespace lgfc{
class fileIO{
    public:
    fileIO() = default;
    fs::path concatenate(fs::path folder, fs::path file){
        // TODO need to add os detection 
        return folder / file;
    }
    fs::path findFileWithin(fs::path& relPath, std::vector<fs::path>& folders){
        for(auto & folder : folders){
            auto abspath = concatenate(folder, relPath);
            if( fs::exists(abspath) ){
                return fs::path(abspath);
            }
        }
        return "";
    }
    fs::path getFile(std::string pathname){
        auto path = fs::absolute(pathname);
        if(!fs::exists(path)) return "";
        inputPaths.push_back(path.parent_path());
        return path;
    }

    void addIncludePath(std::string p){
        includePaths.push_back(fs::absolute(p));
    }
    fs::path findInclude(std::string file){
        return findFileWithin(fs::path(file), includePaths);
    }
    std::vector<fs::path> includePaths;
    std::vector<fs::path> inputPaths;
};

}
#endif