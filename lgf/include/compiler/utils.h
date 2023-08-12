
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

class option {
    public:
    option(option&) = delete;
    option(option&&) = delete;
    static option& get(){
        if(!opt_ptr) opt_ptr = new option();
        return *opt_ptr;
    }
    bool log_lv_trace = 0;
    
    private:
    option() = default;
    inline static option *opt_ptr=nullptr;
};

class trace_logger {
    public:
    trace_logger(const char* curFuncName): funcName(curFuncName){
        active = option::get().log_lv_trace;
        if(active)
            std::cout<<"[--trace--]: Enter "<<funcName<<"..."<<std::endl;
    }
    ~trace_logger(){
        if(active)
            std::cout<<"[--trace--]: Done  "<<funcName<<"."<<std::endl;
    }
    const char* funcName;
    bool active = 0;
};
};

#define TRACE_LOG trace_logger __loger(__func__)


#endif