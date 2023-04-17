

#ifndef GLOBAL_SET_H_
#define GLOBAL_SET_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
namespace global{
class stream
{
public:
    // makes the singleton to be unassignable / non-clonable
    stream(stream &) = delete;
    ~stream() = default;    
    void operator=(const stream &) = delete;
    void liveStreamToConsole() {
        outputf.close();
        os = &std::cout;
    }
    void liveStreamToFile(const std::string& filename) {
        outputf.open(filename);
        os = &outputf;
    }
    void streamToBuffer(){
        outputf.close();
        os=&ss;
    }
    std::string getBufferContent(){
        return ss.str();
    }
    void dumpToConsole(){
        std::cout<<getBufferContent()<<std::endl;
    }
    void dumpToFile(const std::string& filename){
        liveStreamToFile(filename);
        *os<<getBufferContent();
        outputf.close();
    }
    static stream &getInstance(){
        if (gstream != nullptr)
            return *gstream;
        gstream = new stream();
        return *gstream;
    }
    template<typename T>
    stream& operator<<(const T& data) {
        *os << data;
        return *this;
    }
    void printIndent(){
        for (int i = 0; i < curIndentLevel; i++)
            *os << "  ";
    }
    void incrIndentLevel(int n =1){ curIndentLevel+=n;}
    void decrIndentLevel(int n =1){ 
        curIndentLevel-=n;
        if(curIndentLevel<0) curIndentLevel = 0;
    }
    
protected:
    stream() = default;
    
    inline static stream *gstream=nullptr;
    std::ostream* os = &std::cout;
    std::stringstream ss;
    std::ofstream outputf;
    int curIndentLevel=0;
};
};

#endif