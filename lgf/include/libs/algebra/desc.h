
#ifndef LGF_LIB_ALGEBRA_DESC_H
#define LGF_LIB_ALGEBRA_DESC_H

#include "lgf/attribute.h"
#include "lgf/context.h"
#include "lgf/value.h"
#include "lgf/attribute.h"
#include "lgf/utils.h"

namespace lgf
{

  class algebraDesc
  {
  public:
    algebraDesc() = default;
    virtual bool unit_effective_check(valueDesc *desc) = 0;
    virtual bool zero_effective_check(valueDesc *desc) = 0;
  };

  class varDesc : public simpleValue, public algebraDesc
  {
  public:
    varDesc(LGFContext *ctx) : simpleValue("variable") {}
    virtual bool unit_effective_check(valueDesc *desc) final
    {
      return dynamic_cast<varDesc *>(desc) != nullptr;
    }
    virtual bool zero_effective_check(valueDesc *desc) final
    {
      return dynamic_cast<varDesc *>(desc) != nullptr;
    }
  };

  class realNumberAttr : public dataAttr
  {
  public:
    enum numberType : int8_t
    {
      real,
      inf,
      ninf,
      e,
      pi,
      finite
    };
    realNumberAttr(double d) : dataAttr("realNumbere"), data(d) {}
    realNumberAttr(numberType t) : dataAttr("realNumber"), type(t) {}
    // realNumberAttr(realNumberAttr &&attr) : dataAttr("realNumber"), data(attr.get_data()), type(attr.get_type()) {}
    void set_type(numberType t) { type = t; }

    virtual sid_t represent() override
    {
      if (type == real)
        return utils::to_string(data);
      if (type == inf)
        return "inf";
      if (type == ninf)
        return "-inf";
      if (type == e)
        return "e";
      if (type == pi)
        return "pi";
      return "finite";
    }
    double get_data() { return data; }
    numberType get_type() { return type; }
    static realNumberAttr get_pinf()
    {
      realNumberAttr res(0);
      res.set_type(inf);
      return res;
    }
    static realNumberAttr get_ninf()
    {
      realNumberAttr res(0);
      res.set_type(ninf);
      return res;
    }

  private:
    numberType type = real;
    double data = 0;
  };

  class realNumber : public simpleValue, public algebraDesc
  {
  public:
    realNumber(LGFContext *ctx) : simpleValue("realNumber") {}
    virtual bool unit_effective_check(valueDesc *desc) final
    {
      return dynamic_cast<realNumber *>(desc) != nullptr;
    }
    virtual bool zero_effective_check(valueDesc *desc) final
    {
      return dynamic_cast<realNumber *>(desc) != nullptr;
    }
  };

  class unitDesc : public valueDesc
  {
  public:
    unitDesc(LGFContext *ctx, valueDesc *elem) : valueDesc("unit"), elemDesc(elem) {}
    virtual sid_t represent() override
    {
      return "unit<" + elemDesc->represent() + ">";
    }
    valueDesc *get_elem_desc() { return elemDesc; }
    virtual bool unit_effective_check(valueDesc *desc);

  private:
    valueDesc *elemDesc = nullptr;
  };

  class zeroDesc : public valueDesc
  {
  public:
    zeroDesc(LGFContext *ctx, valueDesc *elem) : valueDesc("zero"), elemDesc(elem) {}
    virtual sid_t represent() override
    {
      return "zero<" + elemDesc->represent() + ">";
    }
    valueDesc *get_elem_desc() { return elemDesc; }
    bool zero_effective_check(valueDesc *desc)
    {
      if (auto unit = dynamic_cast<unitDesc *>(desc))
        desc = unit->get_elem_desc();
      else if (auto zero = dynamic_cast<zeroDesc *>(desc))
        desc = zero->get_elem_desc();
      return dynamic_cast<algebraDesc *>(elemDesc)->zero_effective_check(desc);
    }

  private:
    valueDesc *elemDesc = nullptr;
  };

} // namespace lgf

#endif