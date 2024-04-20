#ifndef LGF_BUILTIN_OPS_H_
#define LGF_BUILTIN_OPS_H_
#include "lgf/node.h"
#include "lgf/value.h"
#include "desc.h"
#include "lgf/group.h"
#include <string>
namespace lgf
{
    class moduleOp : public graph
    {
    public:
        moduleOp() : graph("module") {}
        ~moduleOp() {}
        static moduleOp *build(sid_t id)
        {
            auto op = new moduleOp();
            op->set_nontrivial();
            op->name = id;
            return op;
        }
        std::string name = "";
        virtual std::string represent() { return get_sid() + " " + name; }
    };

    class declOp : public node, public normalizer
    {
    public:
        declOp() : node("declOp") {}
        static declOp *build(valueDesc *desc)
        {
            auto op = new declOp();
            op->set_value_desc(desc);
            return op;
        }
        virtual std::string represent()
        {
            printer p;
            p << value_rep() << " = Declare " << value_desc_rep();
            return p.dump();
        }
        virtual resultCode rewrite(painter& p, node *op)
        {
            if (op->get_user_size() == 0)
            {
                op->erase();
                return resultCode::success();
            }
            return resultCode::pass();
        }
    };

    class updateOp : public node
    {
    public:
        updateOp() : node("update") {}
        ~updateOp() {}
        static updateOp *build(node *lhs, node *rhs)
        {
            auto op = new updateOp();
            op->set_value_desc(rhs->get_value_desc());
            op->register_input(lhs, rhs);
            return op;
        }
        node *lhs() { return input(0); }
        node *rhs() { return input(1); }
        virtual std::string represent()
        {
            printer p;
            p << value_rep() << " = " << get_sid() << " : " << input(0)->value_rep() << " by " << input(1)->value_rep();
            return p.dump();
        }
    };

    //----------------------------------------

    class cstDeclOp : public lgf::node
    {
    public:
        cstDeclOp() = default;
        static cstDeclOp *build(valueDesc *data, dataAttr *val)
        {
            auto op = new cstDeclOp();
            op->set_value_desc(data);
            op->set_data(val);
            return op;
        }
        virtual std::string represent()
        {
            printer p;
            p << value_rep() << " = "
              << "Constant: " << value_desc_rep() << " = " << data->represent();
            return p.dump();
        }
        void set_data(dataAttr *val)
        {
            data = val;
        }
        dataAttr *data = nullptr;
    };

    class returnOp : public node
    {
    public:
        returnOp() = default;
        static returnOp *build(node *val)
        {
            auto op = new returnOp();
            op->register_input(val);
            op->set_nontrivial();
            return op;
        }
        virtual std::string represent()
        {
            std::string res = "return";
            res = res + " " + input(0)->value_rep();
            return res;
        }
    };

    // class funcDefineOp : public graph
    // {
    // public:
    //     funcDefineOp() : graph("funcDefineOp") {}
    //     static funcDefineOp *build(std::string id_, valueDesc* ret = nullptr)
    //     {
    //         auto op = new funcDefineOp();
    //         op->set_nontrivial();
    //         op->id = id_;
    //         if(!ret) op->set_desc(ret);
    //         return op;
    //     }
    //     void registerArg(type_t type, std::string id)
    //     {
    //         getEntry().createValue(type, id);
    //     }
    //     value *getCallee() { return outputValue(1); }
    //     value *argument(int n) { return getEntry().outputValue(n + 1); }
    //     std::string id;
    //     type_t getLReturnType() { return returnType; }
    //     virtual std::string represent()
    //     {
    //         printer p;
    //         p << representOutputs() << " = func ";
    //         if (isAbstract)
    //             p << "Register";
    //         else
    //             p << "Def";
    //         p << " : " << id << " (";
    //         p << getEntry().representOutputs() << ")";
    //         if (returnType.getDesc())
    //             p << " -> " << returnType.represent();
    //         return p.dump();
    //     }
    //     bool isAbstract = 1;
    //     lgf::type_t returnType;
    //     virtual void print()
    //     {
    //         global::stream::getInstance().printIndent();
    //         std::string code = represent();
    //         // add space if the represent is not empty
    //         // {} no reprsent, shoudn't have space
    //         // module {}, have represent "module", should have space
    //         // between "module" and the {}.
    //         global::stream::getInstance() << represent();
    //         if (!isAbstract)
    //         {
    //             printGraph();
    //         }
    //         else
    //             global::stream::getInstance() << "\n";
    //     }
    // };

    // class funcCallOp : public node
    // {
    // public:
    //     funcCallOp() = default;
    //     static funcCallOp *build(LGFContext *ctx, value *callee)
    //     {
    //         auto op = new funcCallOp();
    //         op->setNontrivial();
    //         // op->funPtr = callee;
    //         op->registerInput(callee);
    //         auto &ret = callee->getDefiningOp<funcDefineOp>()->returnType;
    //         if (ret.getDesc())
    //         {
    //             op->hasReturn = 1;
    //             op->createValue(ret, "");
    //         }
    //         return op;
    //     }
    //     template <typename... ARGS>
    //     static funcCallOp *build(LGFContext *ctx, value *callee, ARGS... args)
    //     {
    //         auto op = build(ctx, callee);
    //         op->registerInput(args...);
    //         return op;
    //     }
    //     void addArg(value *arg)
    //     {
    //         registerInput(arg);
    //     }
    //     void addArgs(std::vector<value *> &vec)
    //     {
    //         for (auto arg : vec)
    //         {
    //             registerInput(arg);
    //         }
    //     }
    //     value *getCallee() { return inputValue(0); }
    //     value *arg(int n = 0) { return inputValue(n + 1); }
    //     value *returnValue() { return outputValue(1); }
    //     type_t getReturnType()
    //     {
    //         return returnValue()->getType();
    //     }
    //     virtual std::string represent()
    //     {
    //         printer p;
    //         auto callee = getCallee()->getDefiningOp<funcDefineOp>();
    //         if (hasReturn)
    //             p << representOutputs() << " = ";
    //         p << "call " << getCallee()->represent() << " @ " << callee->id << "( ";
    //         if (getInputSize() > 1)
    //         {
    //             p << arg(0)->represent();
    //             for (auto i = 1; i < getInputSize() - 1; i++)
    //             {
    //                 p << ", " << arg(i)->represent();
    //             }
    //         }
    //         p << " )";
    //         return p.dump();
    //     }
    //     bool hasReturn = 0;
    //     value *funPtr = nullptr;
    // };

}
#endif