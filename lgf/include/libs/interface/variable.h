
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

        node* node() const{
            return v;
        }
        
        void operator=(const variable &other)
        {
            v = other.v;
        }

        template <typename opTy, typename daTy>
        variable data_rhs_binary_op(const variable &var, const daTy &data)
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<daTy>(data));
            auto res = canvas::get().get_painter().paint<opTy>(var.node(), cst);
            return variable(res);
        }

        template <typename opTy, typename daTy>
        variable data_lhs_binary_op(const daTy &data, const variable &var) const
        {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<daTy>(data));
            auto res = canvas::get().get_painter().paint<opTy>(cst, var.node());
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
            return var.data_lhs_binary_op<lgf::sumOp, lgf::doubleData>(num, var);
        }
        
        friend variable operator+(const int &num, const variable &var)
        {
            return var.data_lhs_binary_op<lgf::sumOp, lgf::intData>(num, var);
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
            return var.data_lhs_binary_op<lgf::productOp, lgf::doubleData>(num, var);
        }

        friend variable operator*(const int &num, const variable &var)
        {
            return var.data_lhs_binary_op<lgf::productOp, lgf::intData>(num, var);
        }

        variable operator-() const
        {
            auto &ctx = canvas::get().get_context();
            auto res = canvas::get().get_painter().paint<lgf::negativeOp>(v);
            return variable(res);
        }

        variable operator - (const double &num)
        {
            return *this + (-num);
        }

        variable operator - (const int &num)
        {
            return *this + (-num);
        }

        friend variable operator-(const double &num, const variable &var)
        {
            return (-var) + num;
        }
        friend variable operator-(const int &num, const variable &var)
        {
            return (-var) + num;
        }

        variable operator/(const variable &other)
        {
            auto &ctx = canvas::get().get_context();
            auto inv = canvas::get().get_painter().paint<lgf::inverseOp>(other.node());
            auto res = canvas::get().get_painter().paint<lgf::productOp>(v, inv);
            return variable(res);
        }
        
        template<typename T, typename attrT>
        variable binary_data_lhs_divide(const T &num, const variable &rhs) const {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<attrT>(num));
            auto inv = canvas::get().get_painter().paint<lgf::inverseOp>(rhs);
            auto res = canvas::get().get_painter().paint<lgf::productOp>(cst, inv);
            return variable(res);
        }

        template<typename T, typename attrT>
        variable binary_data_rhs_divide(const variable &lhs, const T &num) const {
            auto &ctx = canvas::get().get_context();
            auto real = ctx.get_desc<lgf::realNumber>();
            auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(real, ctx.get_data_attr<attrT>(num));
            auto inv = canvas::get().get_painter().paint<lgf::inverseOp>(cst);
            auto res = canvas::get().get_painter().paint<lgf::productOp>(lhs, inv);
            return variable(res);
        }

        friend variable operator/(const double &num, const variable &var)
        {
            return var.binary_data_lhs_divide<double, lgf::doubleData>(num, var);
        }

        friend variable operator/(const int &num, const variable &var)
        {
            return var.binary_data_lhs_divide<int, lgf::doubleData>(num, var);
        }

        variable operator/( const int &num )
        {
            return binary_data_rhs_divide<int, lgf::intData>(*this, num);
        }

        variable operator/( const double &num)
        {
            return binary_data_rhs_divide<double, lgf::doubleData>(*this, num);
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