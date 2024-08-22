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
        }
        load_input_file(options["_1"], !recursive);
    }
    bool recursive = false;
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
        factory.run(input.string());
    }
    return 0;
}