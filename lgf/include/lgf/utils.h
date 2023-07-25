
#ifndef LGF_UTILS_H
#define LGF_UTILS_H

namespace lgf::utils{

constexpr bool isalpha(unsigned ch) { return (ch | 32) - 'a' < 26; }
constexpr bool isdigit(unsigned ch) { return (ch - '0') < 10; }
constexpr bool isalnum(unsigned ch) {
  return isalpha(ch) || isdigit(ch);
}
constexpr bool isgraph(unsigned ch) { return 0x20 < ch && ch < 0x7f; }
constexpr bool islower(unsigned ch) { return (ch - 'a') < 26; }
constexpr bool isupper(unsigned ch) { return (ch - 'A') < 26; }
constexpr bool isspace(unsigned ch) {
    return ch == ' ' || (ch - '\t') < 5;
}

}
#endif