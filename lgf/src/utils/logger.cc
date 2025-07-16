#include "logger.h"

namespace utils
{
    logStream::logStream(const logLevel level) : streamLevel(level)
    {
        should_stream = streamLevel >= logger::get().get_log_level();
    }
    logStream::~logStream()
    {
        if (should_stream)
        {
            std::lock_guard<std::mutex> lock(logger::get().get_mutex());
            if (streamLevel >= logLevel::LOG_ERROR)
            {
                std::cerr << oss.str() << std::endl;
            }
            else
            {
                std::cout << oss.str() << std::endl;
            }
        }
    }

    // initialize the logger once flag;
    std::once_flag logger::m_flag;

    logger::logger() : lv(LOG_WARNING) {}

    logger &logger::get()
    {
        static logger instance;
        return instance;
    }
} // namespace toolkit