
#ifndef LGF_LIBS_CODEGEN_H
#define LGF_LIBS_CODEGEN_H

#include "ast/ast.h"
#include "parser.h"
#include <filesystem>
#include "utils/stream.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include "writer.h"
#include "ast/context.h"
namespace lgf::codegen
{
    class lgfOpCodeGen
    {
    public:
        lgfOpCodeGen() {}
        ~lgfOpCodeGen() {}
        bool is_valid_file(const std::string &filePath)
        {
            return std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath);
        }
        void output_to_file(const std::string &filePath)
        {
            os.live_stream_to_file(filePath);
        }
        bool search_file(const std::string &folderPath, const std::string &fileName)
        {
            for (const auto &entry : std::filesystem::directory_iterator(folderPath))
            {
                if (is_valid_file(entry.path().string()))
                {
                    if (entry.path().filename() == fileName)
                    {
                        // File found
                        return true;
                    }
                }
                else if (std::filesystem::is_directory(entry.path()))
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
            p.set_input_stream(fs);
            // Enable ANSI color in windows terminal
#ifdef _WIN32
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            DWORD consoleMode;
            GetConsoleMode(hConsole, &consoleMode);
            consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hConsole, consoleMode);
#endif
            std::cout << "\033[33m[ Parsing ]: \033[0m" << filePath << std::endl;
            if (p.parse(ctx))
            {
                std::cerr << "Error: Parsing failed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "     \033[33m parsing finished, generating code... \033[0m" << std::endl;
            if (wm.process(*p.root, os))
            {
                std::cerr << "Error: Code generation failed." << std::endl;
                std::exit(EXIT_FAILURE);
            }
            std::cout << "\033[32m[   Done  ] \033[0m" << std::endl;
        }

    private:
        ::utils::cgstream stream;
        ::utils::fiostream fs;
        ::utils::cgstream os;
        codegen::codegenParser p;
        codegen::writerManager wm;
        ::ast::context ctx;
    };
};

#endif