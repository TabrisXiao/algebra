#ifndef UTILS_LOGGER_H
#define UTILS_LOGGER_H
#include <string>
#include <sstream>
#include <iostream>
#include <mutex>
#include <map>

namespace utils
{
    enum logLevel
    {
        LOG_DEBUG = 0,
        LOG_INFO,
        LOG_WARNING, // default
        LOG_ERROR,
        LOG_FATAL
    };

    static const std::map<logLevel, std::string> logLevelNames = {
        {LOG_DEBUG, "DEBUG"},
        {LOG_INFO, "INFO"},
        {LOG_WARNING, "WARNING"},
        {LOG_ERROR, "ERROR"},
        {LOG_FATAL, "FATAL"}};

    // stream object for logging, it will generate log output in destructor
    class logStream
    {
    public:
        explicit logStream(logLevel level);
        ~logStream();
        template <typename T>
        logStream &operator<<(const T &msg)
        {
            oss << msg;
            return *this;
        }

        logStream &operator<<(std::ostream &(*manip)(std::ostream &))
        {
            manip(oss);
            return *this;
        }

    private:
        logLevel streamLevel;
        std::ostringstream oss;
        bool should_stream; // whether to stream the log, can be set by logger
    };
    class logger
    {
    public:
        logLevel get_log_level() const
        {
            return lv;
        }
        void set_level(logLevel lev)
        {
            lv = lev;
        }
        std::mutex &get_mutex()
        {
            return mtx;
        }
        static logger &get();

        logStream &log(logLevel level = LOG_WARNING)
        {
            return *(new logStream(level));
        }

    private:
        logger();
        ~logger() = default;
        logLevel lv;
        std::mutex mtx;               // thread-safe output
        static std::once_flag m_flag; // ensure single instance
    };

} // namespace toolkit

#endif // UTILS_LOGGER_H