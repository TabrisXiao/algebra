
#ifndef LGF_INTERFACE_VAR_H
#define LGF_INTERFACE_VAR_H
#include "canvas.h"
#include "libs/Builtin/Builtin.h"
#include "libs/algebra/algebra.h"

namespace lgi
{

    class variable
    {
    public:
        variable(bool init = 1)
        {
            if (!init)
                return;
            auto &ctx = canvas::get().get_context();
            v = canvas::get().get_painter().paint<lgf::declOp>(ctx.get_desc<varDesc>());
        }

        variable(lgf::node *val)
        {
            v = val;
        }

        variable(const variable &other)
        {
            v = other.v;
        }
        void operator=(const variable &other)
        {
            v = other.v;
        }

        template <typename opTy, typename daTy>
        variable data_rhs_binary_op(const variable &, const daTy &data)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<daTy>(data));
            auto res = canvas::get().get_painter().paint<opTy>(v, cst);
            return variable(res);
        }

        template <typename opTy, typename daTy>
        variable data_lhs_binary_op(const daTy &data, const variable &)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<daTy>(data));
            auto res = canvas::get().get_painter().paint<opTy>(cst, v);
            return variable(res);
        }

        variable operator+(const double &num)
        {
            return data_rhs_binary_op<lgf::sumOp, lgf::doubleData>(*this, num);
        }

        variable operator+(const int &num)
        {
            return data_rhs_binary_op<lgf::sumOp, lgf::intData>(*this, num);
        }

        friend variable operator+(const double &num, const variable &var)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<doubleData>(num));
            auto res = canvas::get().get_painter().paint<sumOp>(cst, var.v);
            return variable(res);
        }
        
        friend variable operator+(const int &num, const variable &var)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<intData>(num));
            auto res = canvas::get().get_painter().paint<sumOp>(cst, var.v);
            return variable(res);
        }

        variable operator+(const variable &other)
        {
            auto &ctx = canvas::get().get_context();
            auto res = canvas::get().get_painter().paint<lgf::sumOp>(v, other.v);
            return variable(res);
        }

        variable operator*(const variable &other)
        {
            auto &ctx = canvas::get().get_context();
            auto res = canvas::get().get_painter().paint<lgf::productOp>(v, other.v);
            return variable(res);
        }

        variable operator*(const double &num)
        {
            return data_rhs_binary_op<lgf::productOp, lgf::doubleData>(*this, num);
        }

        variable operator*(const int &num)
        {
            return data_rhs_binary_op<lgf::productOp, lgf::intData>(*this, num);
        }

        friend variable operator*(const double &num, const variable &var)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<doubleData>(num));
            auto res = canvas::get().get_painter().paint<productOp>(cst, var.v);
            return variable(res);
        }

        friend variable operator*(const int &num, const variable &var)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<intData>(num));
            auto res = canvas::get().get_painter().paint<productOp>(cst, var.v);
            return variable(res);
        }

        lgf::value &value() const
        {
            return v->get_value();
        }

    protected:
        lgf::node *v = nullptr;
    };

} // namespace lgi

#endif