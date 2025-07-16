#include "utils/logger.h"

#define GLOG(level) utils::logStream(level) << "[" << utils::logLevelNames.at(level) << "]: "

#define THROW(msg) throw std::runtime_error(msg)
