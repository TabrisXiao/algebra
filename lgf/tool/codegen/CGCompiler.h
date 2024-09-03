#ifndef CODEGEN_COMPILER_H
#define CODEGEN_COMPILER_H

#include "CGParser.h"
#include "CGWriter.h"
#include <filesystem>
#include "dependency.h"
namespace codegen
{
    namespace sfs = std::filesystem;
    class CGCompiler : public aoc::fiostream
    {
    public:
        CGCompiler() = default;
        ~CGCompiler() = default;

        void compile(sfs::path input, sfs::path output, sfs::path projectBase)
        {
            CGContext ctx;
            auto basePathStr = projectBase.filename().string();
            std::cout << "\033[33m[ Parsing ]: \033[0m " << input.string() << "  ...  \n";
            auto root = std::move(parse(&ctx, input));
            std::cout << "\033[33m[ writing ]: \033[0m  ...  \n";
            CGWriter w;
            // ctx.print();
            auto obuffer = w.convert(&ctx, root.get(), &depMap, basePathStr);
            write_string_buffer_to_file(output.string().c_str(), obuffer);
            std::cout << "\033[32m[   Done  ] \033[0m: exported: " << output.string() << std::endl;
        }

        std::unique_ptr<astContext> parse(CGContext *ctx, sfs::path input);
        sfs::path parse_import(CGContext *c, CGParser *p);

        void parse_file(CGContext *c, CGParser *p, astContext *root);
        void include(sfs::path p)
        {
            includes.push_back(p);
        }
        void pop_back_inlcude()
        {
            includes.pop_back();
        }
        sfs::path search_file(const sfs::path &path)
        {
            for (auto &it : includes)
            {
                auto p = it / path;
                if (sfs::exists(p))
                {
                    return p;
                }
            }
            return sfs::path();
        }

    private:
        std::vector<sfs::path> includes;
        dependencyMap depMap;
    };
}

#endif