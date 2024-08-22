#include <iostream>
#include "codegen/codegen.h"
#include <optional>
#include "utils/config.h"

class config_t : public utils::cppAppConfig
{
public:
    config_t(int argc, char *argv[])
    {
        load_args(argc, argv);
        parse_args();
    }
    void parse_args()
    {
        parse_inputs();
        if (!has("_1"))
        {
            std::cerr << "Error: No input file provided" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (has("r"))
        {
            recursive = true;
            if (!has("_2"))
            {
                std::cerr << "Error: No output folder provided for recursive run." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
        load_input_file(options["_1"], !recursive);
        if (has("_2"))
        {
            output_path = options["_2"];
            if (recursive && !sfs::is_directory(output_path) && std::filesystem::exists(output_path))
            {
                std::cerr << "Error: Output path must be a directory when using recursive run." << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }
    std::string generate_output_path(const std::string ipath, const std::string opath, const std::string extension)
    {
        if (!recursive)
        {
            std::cerr << "Error: Recursive search not enabled." << std::endl;
        }
        auto path = (sfs::path(opath) / sfs::path(ipath).lexically_relative(options["_1"]).replace_extension(extension)).string();
        create_directories_if_not_exists(path);
        return path;
    }
    bool recursive = false;
    sfs::path output_path = "";
};

int main(int argc, char *argv[])
{
    // all possible options are started with a dash
    config_t config(argc, argv);
    if (config.has("h"))
    {
        std::cout << "Usage: codegen [options] <input_file> <output_file>" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "-r: Recursively search for input files from given path." << std::endl;
        std::cout << "-h: Display this help message." << std::endl;
        std::exit(EXIT_SUCCESS);
    }
    lgf::codegen::lgfOpCodeGen factory;

    for (auto &input : config.input_files)
    {
        auto ofile = config.output_path.string();
        if (config.recursive)
        {
            ofile = config.generate_output_path(input.string(), config.output_path.string(), ".h");
        }
        if (!ofile.empty())
        {
            factory.output_to_file(ofile);
        }
        factory.run(input.string());
    }
    return 0;
}