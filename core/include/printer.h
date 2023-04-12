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