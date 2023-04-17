#ifndef PRINTER_H
#define PRINTER_H
#include <string>
#include <iostream>
#include <sstream>

class printer {
    public :
    printer () = default;
    std::string dump(){
        auto ret = oss.str();
        flush();
        return ret;
    }
    void flush(){oss.str("");}

    template<typename T>
    printer& operator<<(const T& data) {
        oss << data;
        return *this;
    }
    std::ostringstream oss;
};

// utility to help making representation
class coder {
    public : 
    coder() = default;

};

class parser {
    public:
    parser() = default;
    void handle(std::string & str){buffer = str;}
    char getNextChar(){
        if(buffer.size() == curCol-1) return EOF;
        return buffer[curCol++];
    }
    // word is string separated by space like xxx yy are two words.
    // return the next word in the buffer
    std::string nextWord(){
        char lastchar = '\0';
        std::string word; 
        while(lastchar!=' '){
            lastchar = getNextChar();
            if(lastchar!=' ') word = word + lastchar;
        }
        return word;
    }
    std::string getRestBuffer(){return buffer.substr(curCol);}
    std::string buffer;
    int curCol = 0;
};

#endif