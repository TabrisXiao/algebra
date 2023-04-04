#ifndef PRINTER_H
#define PRINTER_H
#include <string>

class printer {
    public :
    printer () = default;
    std::string getString(){
        return buffer;
    }
    void flush(){buffer ="";}
    void addToken(std::string tok){
        if(buffer.empty()) buffer = tok;
        else buffer = buffer+" "+tok;
    }
    void addString(std::string str){
        buffer = buffer+str;
    }
    std::string buffer;
};

#endif