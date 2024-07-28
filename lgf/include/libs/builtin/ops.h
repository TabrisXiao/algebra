#ifndef LGF_BUILTIN_OPS_H_
#define LGF_BUILTIN_OPS_H_
#include "lgf/node.h"
#include "lgf/value.h"
#include "desc.h"
#include "lgf/pass.h"
#include <string>
namespace lgf
{
    enum builtin_feature_e : uint8_t
    {
        eNonTrivial = 0,
        eIdenticalRemovable = 1,
    };

    class moduleOp : public graph
    {
    public:
        moduleOp() : graph("module") {}
        ~moduleOp() {}
        static moduleOp *build(LGFContext *ctx, sid_t id)
        {
            auto op = new moduleOp();
            op->mark_status(eNonTrivial);
            op->name = id;
            return op;
        }
        std::string name = "";
        virtual std::string represent() { return get_sid() + " " + name; }
    };

    class declOp : public node, public normalizer, public identiferInterface
    {
    public:
        declOp() : node("declOp") {}
        static declOp *build(LGFContext *ctx, descriptor &desc)
        {
            auto op = new declOp();
            op->set_value_desc(desc);
            op->mark_status(eNonTrivial);
            return op;
        }
        template <typename... ARGS>
        static declOp *build(LGFContext *ctx, descriptor &desc, ARGS... args)
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
        virtual resultCode normalize(painter &p, node *op)
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

    class cstDeclOp : public lgf::node, public identiferInterface
    {
    public:
        cstDeclOp() { mark_status(eIdenticalRemovable); };
        static cstDeclOp *build(LGFContext *ctx, descriptor &data, attribute &val)
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
            if (!data.is_null())
            {
                p << " = " << data.represent();
            }
            return p.dump();
        }
        void set_data(attribute val)
        {
            data = val;
        }
        attribute &get_data_attr()
        {
            return data;
        }
        attribute data;
    };

    class returnOp : public node
    {
    public:
        returnOp() = default;
        static returnOp *build(LGFContext *ctx, node *val)
        {
            auto op = new returnOp();
            op->register_input(val);
            op->mark_status(eNonTrivial);
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
        static getListElemOp *build(LGFContext *ctx, descriptor elemDesc, node *linput, size_t idx)
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
            auto list = op->input()->get_value_desc().dyn_cast<listDesc>();
            if (!list)
            {
                throw std::runtime_error("getListElemOp: input is not a list");
            }
            if (idx >= list->size())
            {
                throw std::runtime_error("getListElemOp: index out of the input list range");
            }
            // op->set_value_desc(list->get(idx));
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
        funcCallOp()
        {
            mark_status(eNonTrivial);
        }
        static funcCallOp *build(LGFContext *ctx, node *func)
        {
            auto op = new funcCallOp();
            op->register_input(func);
            auto desc = func->get_value_desc().dyn_cast<funcDesc>();
            THROW_WHEN(!desc, "funcCallOp: calling a non-function object!");
            // op->set_value_desc(desc->get_ret_desc());
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
            auto desc = input(0)->get_value_desc().dyn_cast<funcDesc>();
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
            funcArgOp(sid_t id, descriptor desc) : node("funcArg")
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
            mark_status(eNonTrivial);
        }
        void set_name(sid_t n)
        {
            name = n;
        }

        static funcDefineOp *build(LGFContext *ctx, sid_t n, descriptor desc)
        {
            auto op = new funcDefineOp();
            op->set_name(n);
            int i = 0;
            auto fDesc = desc.dyn_cast<funcDesc>();
            for (auto &desc : fDesc->get_arg_descs())
            {
                std::string id = "%arg" + std::to_string(i);
                op->args.emplace_back(std::make_unique<funcArgOp>(id, desc));
            }
            // op->set_value_desc(fDesc);
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
            p += ") -> " + get_value_desc().dyn_cast<funcDesc>()->get_ret_desc().represent();
            return p;
        }

        node *get_arg(size_t idx)
        {
            return args[idx].get();
        }
        void set_arg_desc(size_t idx, descriptor desc)
        {
            args[idx]->set_value_desc(desc);
        }
        sid_t name;
        std::vector<std::unique_ptr<funcArgOp>> args;
    };

}
#endif