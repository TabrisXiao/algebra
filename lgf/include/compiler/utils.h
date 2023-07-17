
#ifndef LGFCOMPILER_UTILS_H
#define LGFCOMPILER_UTILS_H
#include <memory> 
#include <string>
#include <exception>

#define COMPIELR_THROW(msg)\
    std::cerr<<"Compiler Error: " __FILE__ ":"<< std::to_string(__LINE__)<<msg<<std::endl;\
    std::exit(EXIT_FAILURE);

#define COMPIELR_THROW_WHEN(condition, msg)\
    if (condition){\
        std::cerr<<"Compiler Error: " __FILE__ ":"<< std::to_string(__LINE__)<<msg<<std::endl;\
        std::exit(EXIT_FAILURE);\
    }
#define THROW_WHEN(condition, msg) \
    if (condition) { \
        std::cerr<<"Runtime Error: " __FILE__ ":"<< std::to_string(__LINE__)<< ": "<<msg<<"\n"; \
        std::exit(EXIT_FAILURE); \
    }
    
namespace lgf::compiler{
struct location {
    std::shared_ptr<std::string> file; ///< filename.
    int line;                          ///< line number.
    int col;                           ///< column number.
    std::string string(){
        return (*file)+"("+std::to_string(line)+", "+std::to_string(col)+")";
    }
};
};

#endif