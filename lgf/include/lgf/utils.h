
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

namespace lgf{
class logicResult {
  public:
  logicResult() = delete;
  logicResult(logicResult& res) { value = res; }
  static logicResult success() {
    return logicResult(0);
  }
  static logicResult fail() {
    return logicResult(1);
  }
  bool getValue() const { return value; }
  bool operator ==(const logicResult& a){ return value == a.getValue(); }
  operator bool () const { return value; }
  private:
  logicResult(bool t) { value = t; }
  bool value = 0;
};

// bitCode is a template object to encode type tag into binary type:
// 
template<typename digitType>
class bitCode {
  public:
  using digit_t = digitType;
  bitCode(){}
  bitCode(digitType& val) { value = val; }
  bitCode(bitCode& code ){ value = code.value;}
  
  bitCode shift(size_t val) { value |= 1<<val; return *this; }
  bitCode add(bitCode& val) { value |= val.value; return *this; }
  bool check(digitType val){ 
    return (value & val) == val;
  }
  void reset(){ value = 0; }
  digit_t value=0;
};
}
#endif