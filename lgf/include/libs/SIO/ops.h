#ifndef MATH_SIO_OPS_H
#define MATH_SIO_OPS_H
#include "lgf/node.h"
#include "libs/builtin/builtin.h"
#include "libs/algebra/algebra.h"
#include "lgf/group.h"

namespace lgf::SIO
{
    class exportOp : public node
    {
    public:
        exportOp(std::string name) : node(name)
        {
            set_nontrivial();
        }
        static exportOp *build(node *value)
        {
            exportOp *op = new exportOp("sio::export");
            op->register_input(value);
            return op;
        }
        virtual sid_t represent() override
        {
            return get_sid() + ": " + input()->value_rep();
        }
    };

    class latexExportOp : public exportOp
    {
    public:
        latexExportOp() : exportOp("sio::LatexExporter") {}
        static latexExportOp *build(node *value)
        {
            latexExportOp *op = new latexExportOp();
            op->register_input(value);
            return op;
        }
    };

    class representOp : public node
    {
    public:
        representOp(std::string name) : node(name) {}
    };

    class symbolOp : public representOp
    {
    public:
        symbolOp() : representOp("sio::symbol") {}
        static symbolOp *build(valueDesc *val, std::string sid)
        {
            symbolOp *op = new symbolOp();
            op->get_value().set_sid(sid);
            return op;
        }
        std::string get_symbol()
        {
            return get_value_sid();
        }
        virtual std::string represent() override
        {
            printer p;
            p << value_rep() << " = " << get_sid();
            return p.dump();
        }
    };

    class sumOp : public representOp
    {
    public:
        sumOp() : representOp("sio::sum") {}
        static sumOp *build(valueDesc *desc, std::vector<node *> &args)
        {
            sumOp *op = new sumOp();
            op->register_inputs(args);
            op->set_value_desc(desc);
            return op;
        }
    };

    class scalarProductOp : public representOp
    {
    public:
        scalarProductOp() : representOp("sio::scalarProduct") {}
        static scalarProductOp *build(std::vector<node *> args)
        {
            scalarProductOp *op = new scalarProductOp();
            op->register_inputs(args);
            return op;
        }
    };

    class equalOp : public representOp
    {
    public:
        equalOp() : representOp("sio::equal") {}
        static equalOp *build(lgf::LGFContext, node *lhs, node *rhs)
        {
            equalOp *op = new equalOp();
            op->register_input(lhs, rhs);
            return op;
        }
    };

    class assignOp : public representOp
    {
    public:
        assignOp() : representOp("sio::assign") {}
        static assignOp *build(node *lhs, node *rhs)
        {
            assignOp *op = new assignOp();
            op->register_input(lhs, rhs);
            return op;
        }
    };

    class negativeOp : public representOp
    {
    public:
        negativeOp() : representOp("sio::negative") {}
        static negativeOp *build(node *input)
        {
            negativeOp *op = new negativeOp();
            op->register_input(input);
            return op;
        }
    };

    class funcOp : public representOp
    {
    public:
        funcOp() : representOp("sio::func") {}
        static funcOp *build(std::string sid, std::vector<node *> &args)
        {
            funcOp *op = new funcOp();
            op->register_inputs(args);
            return op;
        }
        void setFuncName(std::string name)
        {
            funcName = name;
        }
        virtual std::string represent()
        {
            printer p;
            p << value_rep() << " = "
              << " sio::func " << funcName << "( " << represent_inputs() << " )";
            return p.dump();
        }
        std::string getFuncName() { return funcName; }
        std::string funcName;
    };

    class partialD : public representOp
    {
    public:
        partialD() : representOp("sio::partialD") {}
        static partialD *build(node *func, node *var)
        {
            partialD *op = new partialD();
            op->register_input(func, var);
            return op;
        }
    };

    class differentialOp : public representOp
    {
    public:
        differentialOp() : representOp("sio::differential") {}
        static differentialOp *build(node *input)
        {
            differentialOp *op = new differentialOp();
            op->register_input(input);
            return op;
        }
    };

}

#endif