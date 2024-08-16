
#ifndef UTILS_CONFIG_H
#define UTILS_CONFIG_H
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <map>
namespace sfs = std::filesystem;
namespace utils
{
    class cppAppConfig
    {
    public:
        // convention of the input options are:
        // --option=value
        // -option  (if no value is provided, it is assumed to be false)
        // args  (no dash, assumed to be positional arguments)
        // the map stores the options in the form of key-value pairs
        // the key for positional arguments is _1, _2, _3, ...
        cppAppConfig() = default;
        ~cppAppConfig() = default;
        void load_args(int narg, char *arg[])
        {
            argc = narg;
            argv = arg;
        }
        void parse_inputs()
        {
            int pos = 1;
            for (int i = 1; i < argc; i++)
            {
                bool got_input = 0;
                std::string str = argv[i];
                if (str.substr(0, 2) == "--")
                {
                    options[str.substr(2, str.find("="))] = str.substr(str.find("=") + 1);
                }
                else if (str.substr(0, 1) == "-")
                {
                    options[str.substr(1)] = "true";
                }
                else
                {
                    std::string key = "_" + std::to_string(pos);
                    options[key] = str;
                    pos++;
                }
            }
        }
        bool has(const std::string &key)
        {
            return options.find(key) != options.end();
        }

        void load_input_file(const std::string &path, bool regular_files_only = true)
        {
            if (is_dir(path) && !regular_files_only)
            {
                std::cerr << "Error: Directory provided, but regular files only requested" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else if (is_dir(path))
            {
                for (const auto &entry : sfs::directory_iterator(path))
                {
                    if (entry.is_regular_file())
                    {
                        input_files.push_back(entry.path());
                    }
                }
            }
            else
            {
                input_files.push_back(path);
            }
        }
        bool is_dir(const std::string &path)
        {
            return sfs::is_directory(path);
        }
        std::vector<sfs::path> input_files;
        std::map<std::string, std::string> options;
        int argc;
        char **argv;
    };

} // namespace utils

#endif