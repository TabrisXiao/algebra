
#ifndef LGF_LIBS_CODEGEN_H
#define LGF_LIBS_CODEGEN_H

#include "ast/ast.h"
#include "parser.h"
#include "stream.h"
#include <filesystem>
#include "ast/stream.h"
namespace fs = std::filesystem;

namespace lgf::codegen
{
    class lgfOpCodeGen
    {
    public:
        lgfOpCodeGen() {}
        ~lgfOpCodeGen() {}
        bool is_valid_file(const std::string &filePath)
        {
            return fs::exists(filePath) && fs::is_regular_file(filePath);
        }
        bool search_file(const std::string &folderPath, const std::string &fileName)
        {
            for (const auto &entry : fs::directory_iterator(folderPath))
            {
                if (is_valid_file(entry.path().string()))
                {
                    if (entry.path().filename() == fileName)
                    {
                        // File found
                        return true;
                    }
                }
                else if (fs::is_directory(entry.path()))
                {
                    // Recursively search subdirectories
                    if (search_file(entry.path().string(), fileName))
                    {
                        // File found in subdirectory
                        return true;
                    }
                }
            }
            // File not found
            return false;
        }
        void run(const std::string &filePath)
        {
            if (!is_valid_file(filePath))
            {
                std::cerr << "Error: Invalid file: " << filePath << std::endl;
                std::exit(EXIT_FAILURE);
            }
            fs.load_file(filePath);
            if (p.parse())
            {
                std::cerr << "Error: Parsing failed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }

    private:
        cgstream stream;
        ast::fiostream fs;
        codegen::codegenParser p;
    };
};

#endif