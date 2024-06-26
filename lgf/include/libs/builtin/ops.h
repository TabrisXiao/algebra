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
        static moduleOp *build(LGFContext *ctx, sid_t id)
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
        static declOp *build(LGFContext *ctx, valueDesc *desc)
        {
            auto op = new declOp();
            op->set_value_desc(desc);
            op->set_nontrivial();
            return op;
        }
        template <typename... ARGS>
        static declOp *build(LGFContext *ctx, valueDesc *desc, ARGS... args)
        {
            auto op = new declOp();
            op->set_value_desc(desc);
            op->register_input(args...);
            return op;
        }
        virtual std::string represent()
        {
            printer p;
            p << value_rep() << " = Declare " << value_desc_rep();
            if (get_input_size() > 0)
            {
                p << " from: " << represent_inputs();
            }
            return p.dump();
        }
        virtual resultCode rewrite(painter &p, node *op)
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
        static updateOp *build(LGFContext *ctx, node *lhs, node *rhs)
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
        static cstDeclOp *build(LGFContext *ctx, valueDesc *data, dataAttr *val = nullptr)
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
              << "Constant: " << value_desc_rep();
            if (data)
            {
                p << " = " << data->represent();
            }
            return p.dump();
        }
        void set_data(dataAttr *val)
        {
            data = val;
        }
        dataAttr *get_data_attr()
        {
            return data;
        }
        dataAttr *data = nullptr;
    };

    class returnOp : public node
    {
    public:
        returnOp() = default;
        static returnOp *build(LGFContext *ctx, node *val)
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

    class getListElemOp : public node
    {
    public:
        getListElemOp() = default;
        static getListElemOp *build(LGFContext *ctx, valueDesc *elemDesc, node *linput, size_t idx)
        {
            auto op = new getListElemOp();
            op->register_input(linput);
            op->index = idx;
            op->set_value_desc(elemDesc);
            return op;
        }
        static getListElemOp *build(LGFContext *ctx, node *linput, size_t idx)
        {
            auto op = new getListElemOp();
            op->register_input(linput);
            op->index = idx;
            auto list = op->input()->get_value_desc()->dyn_cast<listDesc>();
            if (!list)
            {
                throw std::runtime_error("getListElemOp: input is not a list");
            }
            if (idx >= list->size())
            {
                throw std::runtime_error("getListElemOp: index out of the input list range");
            }
            op->set_value_desc(list->get(idx));
            return op;
        }
        size_t get_index() { return index; }

        virtual std::string represent()
        {
            std::string res = value_rep() + " = " + input(0)->value_rep() + " [" + std::to_string(index) + "]";
            return res;
        }
        size_t index;
    };

    class updateListElemOp : public node
    {
    public:
        updateListElemOp() = default;
        static updateListElemOp *build(LGFContext *ctx, node *linput, size_t idx, node *val)
        {
            auto op = new updateListElemOp();
            op->register_input(linput, val);
            op->index = idx;
            op->set_value_desc(val->get_value_desc());
            return op;
        }
        virtual std::string represent()
        {
            std::string res = value_rep() + " = " + input(0)->value_rep() + " [" + std::to_string(index) + "] = " + input(1)->value_rep();
            return res;
        }
        size_t index;
    };

    class funcCallOp : public node
    {
    public:
        funcCallOp() = default;
        static funcCallOp *build(LGFContext *ctx, node *func)
        {
            auto op = new funcCallOp();
            op->register_input(func);
            auto desc = func->get_value_desc()->dyn_cast<funcDesc>();
            THROW_WHEN(!desc, "funcCallOp: calling a non-function object!");
            op->set_value_desc(desc->get_ret_desc());
            return op;
        }
        template <typename... ARGS>
        static funcCallOp *build(LGFContext *ctx, node *func, ARGS... args)
        {
            auto op = build(ctx, func);
            op->register_input(args...);
            return op;
        }
        node *get_func() { return input(0); }
        virtual sid_t represent()
        {
            auto desc = input(0)->get_value_desc()->dyn_cast<funcDesc>();
            sid_t p;
            p = value_rep() + " = " + "call " + input(0)->value_rep() + ": (";
            for (size_t i = 1; i < get_input_size(); i++)
            {
                p += input(i)->value_rep();
                p += ", ";
            }
            p.pop_back();
            p.pop_back();
            p += ")";
            return p;
        }
    };

    class funcDefineOp : public graph
    {
    public:
        class funcArgOp : public node
        {
        public:
            funcArgOp(sid_t id, valueDesc *desc) : node("funcArg")
            {
                get_value().set_sid(id);
                set_value_desc(desc);
            }
            virtual sid_t represent()
            {
                return get_value_sid() + ": " + value_desc_rep();
            }
        };

        funcDefineOp() : graph("define func:")
        {
            set_nontrivial();
        }
        void set_name(sid_t n)
        {
            name = n;
        }

        static funcDefineOp *build(LGFContext *ctx, sid_t n, funcDesc *fDesc)
        {
            auto op = new funcDefineOp();
            op->set_name(n);
            int i = 0;
            for (auto &desc : fDesc->get_arg_descs())
            {
                std::string id = "%arg" + std::to_string(i);
                op->args.emplace_back(std::make_unique<funcArgOp>(id, desc));
            }
            op->set_value_desc(fDesc);
            return op;
        }

        virtual sid_t represent()
        {
            sid_t p;
            p = value_rep() + " = " + get_sid() + " " + name + "(";
            for (auto &arg : args)
            {
                p += arg->represent();
                p += ", ";
            }
            p.resize(p.size() - 2);
            p += ") -> " + get_value_desc()->dyn_cast<funcDesc>()->get_ret_desc()->represent();
            return p;
        }

        node *get_arg(size_t idx)
        {
            return args[idx].get();
        }
        void set_arg_desc(size_t idx, valueDesc *desc)
        {
            args[idx]->set_value_desc(desc);
        }
        sid_t name;
        std::vector<std::unique_ptr<funcArgOp>> args;
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
    //         // op->funPtr = z;
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