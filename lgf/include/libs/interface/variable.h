
#ifndef LGF_INTERFACE_VAR_H
#define LGF_INTERFACE_VAR_H
#include "canvas.h"
#include "libs/builtin/builtin.h"
#include "libs/math/algebra/algebra.h"
#include "libs/sio/ops.h"

namespace lgi
{
  class variableBase
  {
  public:
    variableBase() = default;
    void check() { canvas::get().get_painter().paint<lgf::returnOp>(v); }
    void latex()
    {
      canvas::get().get_painter().paint<lgf::sio::latexExportOp>(v);
    }
    variableBase(const variableBase &other) { v = other.v; }
    variableBase(lgf::node *val) { v = val; }

    void operator=(const variableBase &other) { v = other.v; }

    lgf::node *node() const { return v; }

    lgf::value &value() const { return v->get_value(); }

  protected:
    lgf::node *v = nullptr;
  };

  class variable : public variableBase
  {
  public:
    variable(bool init = 1)
    {
      if (!init)
        return;
      auto &ctx = canvas::get().get_context();
      v = canvas::get().get_painter().paint<lgf::declOp>(
          lgf::varDesc::get());
    }
    virtual ~variable() {}

    variable(lgf::node *val) : variableBase(val) {}

    variable operator=(const double &rhs)
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto data = lgf::realNumberData::get(lgf::realNumberData::real, rhs);
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, data);
      return variable(cst);
    }
    variable operator=(const int &rhs)
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto data = lgf::realNumberData::get(lgf::realNumberData::real, rhs);
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, data);
      return variable(cst);
    }

    variable(const double &rhs)
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto data = lgf::realNumberData::get(lgf::realNumberData::real, rhs);
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, data);
      v = cst;
    }

    variable(const int &rhs)
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto data = lgf::realNumberData::get(lgf::realNumberData::real, rhs);
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, data);
      v = cst;
    }

    template <typename opTy, typename daTy>
    variable data_rhs_binary_op(const variable &var, const daTy &data)
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, lgf::realNumberData::get(lgf::realNumberData::real, data));
      auto res = canvas::get().get_painter().paint<opTy>(var.node(), cst);
      return variable(res);
    }

    template <typename opTy, typename daTy>
    variable data_lhs_binary_op(const daTy &data, const variable &var) const
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, lgf::realNumberData::get(lgf::realNumberData::real, data));
      auto res = canvas::get().get_painter().paint<opTy>(cst, var.node());
      return variable(res);
    }

    variable operator+(const double &num)
    {
      return data_rhs_binary_op<lgf::sumOp, double>(*this, num);
    }

    variable operator+(const int &num)
    {
      return data_rhs_binary_op<lgf::sumOp, int>(*this, num);
    }

    friend variable operator+(const double &num, const variable &var)
    {
      return var.data_lhs_binary_op<lgf::sumOp, double>(num, var);
    }

    friend variable operator+(const int &num, const variable &var)
    {
      return var.data_lhs_binary_op<lgf::sumOp, int>(num, var);
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
      return data_rhs_binary_op<lgf::productOp, double>(*this, num);
    }

    variable operator*(const int &num)
    {
      return data_rhs_binary_op<lgf::productOp, int>(*this, num);
    }

    friend variable operator*(const double &num, const variable &var)
    {
      return var.data_lhs_binary_op<lgf::productOp, double>(num, var);
    }

    friend variable operator*(const int &num, const variable &var)
    {
      return var.data_lhs_binary_op<lgf::productOp, int>(num, var);
    }

    variable operator-() const
    {
      auto &ctx = canvas::get().get_context();
      auto res = canvas::get().get_painter().paint<lgf::negativeOp>(v);
      return variable(res);
    }

    variable operator-(const double &num) { return *this + (-num); }

    variable operator-(const int &num) { return *this + (-num); }

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

    template <typename T, typename attrT>
    variable binary_data_lhs_divide(const T &num, const variable &rhs) const
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, attrT::get(num));
      auto inv = canvas::get().get_painter().paint<lgf::inverseOp>(rhs.node());
      auto res = canvas::get().get_painter().paint<lgf::productOp>(cst, inv);
      return variable(res);
    }

    template <typename T, typename attrT>
    variable binary_data_rhs_divide(const variable &lhs, const T &num) const
    {
      auto &ctx = canvas::get().get_context();
      auto real = lgf::realNumber::get();
      auto cst = canvas::get().get_painter().paint<lgf::cstDeclOp>(
          real, attrT::get(num));
      auto inv = canvas::get().get_painter().paint<lgf::inverseOp>(cst);
      auto res =
          canvas::get().get_painter().paint<lgf::productOp>(lhs.node(), inv);
      return variable(res);
    }

    friend variable operator/(const double &num, const variable &var)
    {
      return var.binary_data_lhs_divide<double, lgf::float32Data>(num, var);
    }

    friend variable operator/(const int &num, const variable &var)
    {
      return var.binary_data_lhs_divide<int, lgf::float32Data>(num, var);
    }

    variable operator/(const int &num)
    {
      return binary_data_rhs_divide<int, lgf::int32Data>(*this, num);
    }

    variable operator/(const double &num)
    {
      return binary_data_rhs_divide<double, lgf::float32Data>(*this, num);
    }
  };

  namespace constant
  {
    variable pi()
    {
      auto &ctx = canvas::get().get_context();
      auto res = canvas::get().get_painter().paint<lgf::cstDeclOp>(lgf::realNumber::get(), lgf::realNumberData::get(lgf::realNumberData::pi));
      return variable(res);
    }
    variable e()
    {
      auto &ctx = canvas::get().get_context();
      auto res = canvas::get().get_painter().paint<lgf::cstDeclOp>(lgf::realNumber::get(), lgf::realNumberData::get(lgf::realNumberData::e));
      return variable(res);
    }

  } // namespace const

} // namespace lgi

#endif