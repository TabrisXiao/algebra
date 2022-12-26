#ifndef LOGGER_H_
#define LOGGER_H_
#include <iostream>

class logger
{
public:
    enum level
    {
        eError,
        eDebug,
        eTrace
    };

    // makes the singleton to be unassignable / non-clonable
    logger(logger &log) = delete;
    void operator=(const logger &) = delete;

    static logger *getInstance(){
        if (glogger != nullptr)
            return glogger;
        glogger = new logger(eError);
        return glogger;
    }

    std::ostream &trace_ostream(const char *func_name)
    {
        std::cout.clear();
        // if the logging level is smaller than the `trace` level, suppress the logging output.
        if (verbose_level < eTrace)
        {
            std::cout.setstate(std::ios_base::badbit);
            return std::cout;
        }
        std::string fname = func_name;
        fname = "---[ " + fname + " ]: ";
        std::cout << fname;
        return std::cout;
    }

    // provide an ostream for error log function `runtimelog_error`, this ostream should be
    // print regardless of the logging verbose level.
    std::ostream &error_ostream(const char *func_name)
    {
        std::cout.clear();
        std::string fname = func_name;
        fname = "! Error on [ " + fname + " ]: ";
        std::cout << fname;
        return std::cout;
    }

    static std::ostream &cout()
    {
        std::cout.clear();
        return std::cout;
    }

    void setVerbose(level lv);
    static void runtime_exception(bool key, std::string message);
    level verboseLevel() { return verbose_level; }

protected:
    logger(level lv)
    {
        verbose_level = lv;
    }
    inline static logger *glogger=nullptr;
    level verbose_level;
};
#endif