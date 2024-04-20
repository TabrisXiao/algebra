
#include "libs/algebra/desc.h"
#include "lgf/value.h"

namespace lgf {
bool unitDesc::unit_effective_check(valueDesc *desc) {
  if (auto unit = dynamic_cast<unitDesc *>(desc))
    desc = unit->get_elem_desc();
  else if (auto zero = dynamic_cast<zeroDesc *>(desc))
    desc = zero->get_elem_desc();
  return dynamic_cast<algebraDesc *>(elemDesc)->unit_effective_check(desc);
}
} // namespace lgf
