
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

// byteCode is a template object to encode type tag into binary type:
// 
template<typename byteType>
class byteCode {
  public:
  using byte_t = byteType;
  byteCode(){}
  byteCode(byteType& val) { value = val; }
  byteCode(byteCode& code ){ value = code.value;}
  
  byteCode shift(byteType& val) { value |= 1<<val; return *this; }
  byteCode add(byteCode& val) { value |= val.value; return *this; }
  bool check(byteType val){ 
    return (value & val) == val;
  }
  void reset(){ value = 0; }
  byte_t value=0;
};
}
#endif