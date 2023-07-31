
#ifndef COMPILER_AST_H
#define COMPILER_AST_H

#include <vector>
#include <string>
#include "lexer.h"
#include "lgf/printer.h"
#include "streamer.h"
#include <optional>

namespace lgf::compiler {

enum astKind{
    kind_module,
    kind_struct,
    kind_varDecl,
    kind_variable,
    kind_funcDecl,
    kind_funcCall,
    kind_number,
    kind_return,
    kind_binary,
};

class astBase {
    public:
    astBase (location loca, astKind k) : kind(k), loc(loca) {}
    virtual ~astBase () {}
    astKind kind;
    location loc;
    virtual void emitIR(lgf::streamer &) = 0;
};

class moduleAST : public astBase {
    public:
    moduleAST(location loc, int id_, int previous_scop_id = 0)
    : astBase(loc, kind_module)
    , id(id_)
    , previous_id(previous_scop_id) {}
    void addASTNode(std::unique_ptr<astBase>&& ptr){
        contents.push_back(std::move(ptr));
    }
    std::vector<std::unique_ptr<astBase>> contents;
    unsigned int id, previous_id=0;
    virtual void emitIR(lgf::streamer & out){
        out.printIndent();
        out<<"module #"<<id<<" {\n";
        out.incrIndentLevel();
        for(auto & op : contents){
            out.printIndent();
            op->emitIR(out);
            out<<"\n";
        }
        out.decrIndentLevel();
        out.printIndent();
        out<<"}\n";
    }
};

class structAST : public astBase {
    public:
    structAST(location loc): astBase(loc, kind_struct){}
    ~structAST(){ contents.clear(); }
    template<typename astNode, typename ...ARGS>
    void createASTNode(ARGS ...args){
        contents.push_back(astNode(args));
    }
    // Overload for accepting no arguments
    template<typename astNode>
    void createASTNode(){
        contents.push_back(astNode());
    }
    std::vector<std::unique_ptr<astBase>> contents;
    virtual void emitIR(lgf::streamer & out){
        //TODO
    }
};

class varDeclAST : public astBase {
    public:
    varDeclAST(location loc, std::string tid, std::string name_) 
    : astBase(loc, kind_varDecl)
    , typeStr(tid)
    , id(name_) {}
    std::string typeStr, id;
    virtual void emitIR(lgf::streamer & out){
        out<<typeStr<<" "<<id;
    }
};

class varAST : public astBase {
    public:
    varAST(location loc, std::string name)
    : astBase(loc, kind_variable)
    , id(name) {}
    std::string id;
    virtual void emitIR(lgf::streamer & out){
        out<<id;
    }
};


class funcDeclAST : public astBase {
    public:
    funcDeclAST(location loc, std::string funid)
    : astBase(loc, kind_funcDecl), funcID(funid) {}
    std::string funcID;
    std::vector<std::unique_ptr<astBase>> args;
    std::vector<std::unique_ptr<astBase>> contents;
    virtual void emitIR(lgf::streamer & out){
        out<<"def "<<funcID<<"(";
        if(args.size()>0) out<<dynamic_cast<varDeclAST*>(args[0].get())->id;
        for(auto i=1; i<args.size(); i++){
            out<<", "<<dynamic_cast<varDeclAST*>(args[i].get())->id;
        }
        out<<")";
        if(!returnTypeStr.empty()){
            out<<" -> "<<returnTypeStr;
        }
    }
    void setReturnType(std::string & str) {
        returnTypeStr = str;
    }
    std::string returnTypeStr = "";
    bool contentDefined = 0;
};

class numberAST : public astBase {
    public:
    numberAST(location loc, double num) 
    : astBase(loc, kind_number)
    , number(num) {}
    double getValue(){return number;}
    bool isInt(){
        return int(number)== number;
    }
    double number;
    virtual void emitIR(lgf::streamer & out){
        out<<std::to_string(number);
    }
};

class binaryAST : public astBase {
    public:
    binaryAST(location loc, 
    std::string op, 
    std::unique_ptr<astBase>& lhs_,
    std::unique_ptr<astBase>& rhs_)
    : astBase(loc, kind_binary)
    , lhs(std::move(lhs_))
    , rhs(std::move(rhs_))
    , binaryOp(op) {}
    std::unique_ptr<astBase> lhs, rhs;
    std::string binaryOp;
    bool irNotEnd = 0;
    virtual void emitIR(lgf::streamer & out){
        lhs->emitIR(out);
        out<<" "<<binaryOp<<" ";
        rhs->emitIR(out);
    }
};

class funcCallAST : public astBase {
    public:
    funcCallAST(location loc) : astBase(loc, kind_funcCall){}
    std::unique_ptr<astBase>& arg(int n=0){
        return args[n];
    }
    std::vector<std::unique_ptr<astBase>> args;
};

class returnAST : public astBase {
    public:
    returnAST(location loc) : astBase(loc, kind_return) {}
    bool hasValue(){
        return value == nullptr;
    }
    void takeValue(std::unique_ptr<astBase>& ptr){
        value=std::move(ptr);
    }
    std::unique_ptr<astBase> value=nullptr;
    virtual void emitIR(lgf::streamer & out){
        out.printIndent();
        out<<"return";
        if(value){
            value->emitIR(out);
        }
        out<<"\n";
    }
};

}

#endif