#ifndef LGF_MATH_H_
#define LGF_MATH_H_

#include "graph/operation.h"

namespace lgf::math
{
    class variable : public descBase
    {
    public:
        variable(const char *name = "math::variable") : descBase(name) {}
        static description get()
        {
            return description::get<variable>();
        }
    };

    class infinitesmal : public descBase
    {
    public:
        infinitesmal(const description &base_desc) : descBase("math::infinitesmal"), bd(base_desc) {}
        static description get(description &type)
        {
            return description::get<infinitesmal>(type);
        }

        description get_base_desc() const
        {
            return bd;
        }

        description bd;
    };

    class function : public descBase
    {
    public:
        function(const char *name = "math::function") : descBase(name) {}
    };

    class sumOp : public operation
    {
    public:
        sumOp(region *r = nullptr) : operation("math::sum", r) {}
        template <typename... ARGS>
        static std::unique_ptr<sumOp> build(region *r, description desc, ARGS... args)
        {
            auto op = std::make_unique<sumOp>(r);
            (op->register_input(args), ...);
            op->create_object(desc);
            return std::move(op);
        }
        virtual std::string represent() override
        {
            std::cout << " output size: " << get_output_size() << std::endl;
            std::string res = output(0).represent() + " = ";
            std::cout << "get_input_size(): " << get_input_size() << std::endl;
            for (size_t i = 0; i < get_input_size(); i++)
            {
                std::cout << " i " << i << std::endl;
                if (i > 0)
                    res += " + ";
                res += input(i).get_sid();
            }
            return res;
        }
        virtual std::unique_ptr<operation> copy() override
        {
            return clone<sumOp>();
        }
    };

    class productOp : public operation
    {
    public:
        productOp(region *r = nullptr) : operation("math::product", r) {}
        template <typename... ARGS>
        static std::unique_ptr<productOp> build(region *r, ARGS... args)
        {
            auto op = std::make_unique<productOp>(r);
            (op->register_input(args), ...);
            return std::move(op);
        }
        virtual std::string represent() override
        {
            std::string res = output(0).represent() + " = ";
            for (size_t i = 0; i < get_input_size(); i++)
            {
                if (i > 0)
                    res += " * ";
                res += input(i).get_sid();
            }
            return res;
        }
        virtual std::unique_ptr<operation> copy() override
        {
            return clone<productOp>();
        }
    };

    class inverseOp : public operation
    {
    public:
        inverseOp(region *r = nullptr) : operation("math::inverse", r) {}
        static std::unique_ptr<inverseOp> build(region *r, object *input)
        {
            auto op = std::make_unique<inverseOp>(r);
            op->create_object(input->get_desc());
            op->register_input(input);
            return std::move(op);
        }
        virtual std::string represent() override
        {
            return output(0).represent() + " = " + get_name() + "(" + input(0).get_sid() + ")";
        }
        virtual std::unique_ptr<operation> copy() override
        {
            return clone<inverseOp>();
        }
    };

    class approximation : public operation
    {
    public:
        approximation(region *r = nullptr) : operation("math::approx", r) {}
        static std::unique_ptr<approximation> build(region *r, object *input, object *var, object *target)
        {
            auto op = std::make_unique<approximation>(r);
            op->create_object(input->get_desc());
            op->register_input(input, var, target);
            return std::move(op);
        }
        virtual std::string represent() override
        {
            return output(0).represent() + " = " + get_name() + " " + input(0).get_sid() + ": " +
                   input(1).get_sid() + " -> " + input(2).get_sid();
        }
        virtual std::unique_ptr<operation> copy() override
        {
            return clone<approximation>();
        }
    };
}

#endif