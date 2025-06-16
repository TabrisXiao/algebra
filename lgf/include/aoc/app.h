
#ifndef UTILS_CONFIG_H
#define UTILS_CONFIG_H
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <map>
#include "stream.h"
#include "object.h"
namespace sfs = std::filesystem;
namespace aoc::app
{
    class queryInfo
    {
    public:
        queryInfo() = default;
        virtual ~queryInfo() = default;
    };

    class oneTimeInterface
    {
    public:
        // this type of interface is used for one-time applications
        // where the user provides a set of queries at once and application
        // processes them and exits
        oneTimeInterface() = default;
        ~oneTimeInterface() = default;
        virtual void parse_query(int narg, char *argv[]) = 0;
        std::map<std::string, std::string> parse_args(int narg, char *argv[])
        {
            std::map<std::string, std::string> options;
            int pos = 1;
            for (int i = 1; i < narg; i++)
            {
                bool got_input = 0;
                std::string str = argv[i];
                if (str.substr(0, 2) == "--")
                {
                    options[str.substr(2, str.find("="))] = str.substr(str.find("=") + 1);
                }
                else if (str.substr(0, 1) == "-")
                {
                    options[str.substr(1)] = "1";
                }
                else
                {
                    std::string key = "_" + std::to_string(pos);
                    options[key] = str;
                    pos++;
                }
            }
            return options;
        }
        std::vector<sfs::path> get_file_list(const stringRef &folder)
        {
            std::vector<sfs::path> files;
            if (sfs::is_regular_file(folder.data()))
            {
                files.push_back(folder.data());
                return files;
            }
            for (const auto &entry : sfs::recursive_directory_iterator(folder.data()))
            {
                if (entry.is_regular_file())
                {
                    files.push_back(entry.path());
                }
            }
            return files;
        }
        sfs::path create_output_file_path(sfs::path ipath, sfs::path inputPath, sfs::path outputPath, const std::string extension)
        {
            // ipath is a file in the folder/subfolder of inputPath,
            // output path is the folder receiving the output files.
            // this function create a relative path from inputPath to ipath
            // and then append it to outputPath so that the output file
            // is in the same relative location as the input file
            if (!sfs::is_directory(outputPath) && sfs::exists(outputPath))
                return outputPath;

            auto path = sfs::path(outputPath) / sfs::path(ipath).lexically_relative(inputPath).replace_extension(extension);
            return path;
        }
        std::vector<std::unique_ptr<queryInfo>> &get_actions() { return actions; }
        template <typename T, typename... ARGS>
        void add_query(ARGS &&...args)
        {
            actions.push_back(std::make_unique<T>(std::forward<ARGS>(args)...));
        }

    private:
        std::vector<std::unique_ptr<queryInfo>> actions;
    };

    class IOModule : public fiostream
    {
    public:
        IOModule() = default;
        ~IOModule() = default;
    };

    class appleCore
    {
    public:
        appleCore() = default;
        ~appleCore() = default;
        virtual void run(queryInfo *q) = 0;
    };

    class oneShotApp
    {
    public:
        oneShotApp() = default;
        ~oneShotApp() = default;
        template <typename T, typename... ARGS>
        void create_interface(ARGS &&...args)
        {
            inter = std::make_unique<T>(std::forward<ARGS>(args)...);
        }
        template <typename T, typename... ARGS>
        void create_io(ARGS &&...args)
        {
            io = std::make_unique<T>(std::forward<ARGS>(args)...);
        }
        template <typename T, typename... ARGS>
        void create_backend(ARGS &&...args)
        {
            c = std::make_unique<T>(std::forward<ARGS>(args)...);
        }
        virtual void process_query(appleCore *, queryInfo *) = 0;
        virtual void init() = 0;
        void run(int narg, char *argv[])
        {
            inter->parse_query(narg, argv);
            for (auto &it : inter->get_actions())
            {
                process_query(c.get(), it.get());
            }
        }

    private:
        std::unique_ptr<oneTimeInterface> inter;
        std::unique_ptr<IOModule> io;
        std::unique_ptr<appleCore> c;
    };

    class config
    {
    public:
        // convention of the input options are:
        // --option=value
        // -option  (if no value is provided, it is assumed to be false)
        // args  (no dash, assumed to be positional arguments)
        // the map stores the options in the form of key-value pairs
        // the key for positional arguments is _1, _2, _3, ...
        config() = default;
        ~config() = default;

        void parse_inputs()
        {
        }
        bool has(const std::string &key)
        {
            return options.find(key) != options.end();
        }

        void load_input_file(std::string &path, bool regular_files_only = true)
        {
            if (!sfs::exists(path))
            {
                std::cerr << "Error: File or directory does not exist: " << path << std::endl;
                std::exit(EXIT_FAILURE);
            }
            if (is_dir(path) && regular_files_only)
            {
                std::cerr << "Error: Directory provided, but regular files only requested" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            else if (is_dir(path))
            {
                for (const auto &entry : sfs::recursive_directory_iterator(path))
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
        void create_directories_if_not_exists(const std::string &path)
        {
            std::filesystem::path dir = sfs::path(path).parent_path();
            if (!std::filesystem::exists(dir))
            {
                if (!std::filesystem::create_directories(dir))
                {
                    std::cerr << "Error: Failed to create directories " << path << std::endl;
                    std::exit(EXIT_FAILURE);
                }
            }
        }
        std::vector<sfs::path> input_files;
        std::map<std::string, std::string> options;
        int argc;
        char **argv;
    };

} // namespace utils

#endif