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

    lgf::codegen::lgfOpCodeGen factory;
    for (auto &input : config.input_files)
    {
        factory.run(input.string());
    }
    return 0;
}