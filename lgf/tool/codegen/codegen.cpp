#include <iostream>
// #include "codegen.h"
#include <optional>
#include "aoc/app.h"
#include "ast/lexer.h"
#include "CGParser.h"
#include "CGWriter.h"
#include "CGContext.h"
#include "CGCompiler.h"
#ifdef _WIN32
#include <windows.h>
#endif

class CGQueryInfo : public aoc::app::queryInfo
{
public:
    CGQueryInfo(const sfs::path &input, const sfs::path &base_folder, const sfs::path &output, size_t id = 0)
        : input(input), output(output), base(base_folder), qid(id) {}

    size_t qid = 0; // represent different query types
    sfs::path input, output, base;
};

class CGInterface : public aoc::app::oneTimeInterface
{
public:
    CGInterface() = default;
    ~CGInterface() = default;

    virtual void parse_query(int narg, char *argv[])
    {
        std::map<std::string, std::string> options = parse_args(narg, argv);
        THROW_WHEN(options.find("_1") == options.end(), "Error: No input file provided");
        auto input = options["_1"];
        auto output = options["_2"];

        auto is_cursive = options.find("r") != options.end();
        THROW_WHEN(sfs::is_directory(output) && !is_cursive, "Eorr: Output path must be a directory when using recursive run.");

        THROW_WHEN(!sfs::exists(input), "Error: File does not exist: " + input);
        auto fileList = get_file_list(aoc::stringRef(input));
        if (options.find("h") != options.end())
        {
            std::cout << "Usage: codegen [options] <input_file> <output_file>" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "-h: Display this help message." << std::endl;
            std::cout << "-r: Recursively search for files in the input directory." << std::endl;
            std::exit(EXIT_SUCCESS);
        }
        for (auto &file : fileList)
        {
            auto ofile = create_output_file_path(file, input, output, ".h");
            add_query<CGQueryInfo>(file, input, ofile);
        }
    }
};

class CGCore : public aoc::app::appleCore, public codegen::CGCompiler
{
public:
    CGCore() {}
    virtual void run(aoc::app::queryInfo *info) override
    {
        auto query = dynamic_cast<CGQueryInfo *>(info);
        auto input = query->input;
        auto output = query->output;
        if (sfs::is_directory(query->base))
        {
            include(query->base);
        }
        else
        {
            include(query->base.parent_path());
        }

        compile(input, output);
        pop_back_inlcude();
    }
};

class codegenApp : public aoc::app::oneShotApp
{
public:
    codegenApp()
    {
        create_interface<CGInterface>();
        create_backend<CGCore>();
    }
    virtual void init() override
    {
    }
    ~codegenApp() = default;
    virtual void process_query(aoc::app::appleCore *c, aoc::app::queryInfo *q) override final
    {
        try
        {
            c->run(q);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
};

int main(int argc, char *argv[])
{
    // Enable ANSI color in windows terminal
#ifdef _WIN32
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD consoleMode;
    GetConsoleMode(hConsole, &consoleMode);
    consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    SetConsoleMode(hConsole, consoleMode);
#endif

    // all possible options are started with a dash
    codegenApp cg;
    cg.run(argc, argv);

    return 0;
}