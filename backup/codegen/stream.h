

#ifndef CODEGEN_STREAM_H_
#define CODEGEN_STREAM_H_

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

namespace lgf{
namespace codegen{
class cgstream
{
public:
    cgstream() = default;
    // makes the singleton to be unassignable / non-clonable
    //stream(stream &) = delete;
    ~cgstream() = default; 
    //void operator=(const stream &) = delete;
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
    template<typename T>
    cgstream& operator<<(const T& data) {
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
    
    std::ostream* os = &std::cout;
    std::stringstream ss;
    std::ofstream outputf;
    int curIndentLevel=0;
};

class indentGuard {
    public:
    indentGuard() = delete;
    indentGuard(cgstream& sg){
        st = &sg;
        st->incrIndentLevel();
    }
    ~indentGuard(){
        st->decrIndentLevel();
    }
    cgstream *st=nullptr;
};
} //namespace codegen
} //namespace lgf

#endif