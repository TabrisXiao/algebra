
#ifndef LGF_UTILS_H
#define LGF_UTILS_H
#include <optional>
#include <string>
#include <iostream>
namespace lgf::utils
{

  constexpr bool isalpha(unsigned ch) { return (ch | 32) - 'a' < 26; }
  constexpr bool isdigit(unsigned ch) { return (ch - '0') < 10; }
  constexpr bool isalnum(unsigned ch)
  {
    return isalpha(ch) || isdigit(ch);
  }
  constexpr bool isgraph(unsigned ch) { return 0x20 < ch && ch < 0x7f; }
  constexpr bool islower(unsigned ch) { return (ch - 'a') < 26; }
  constexpr bool isupper(unsigned ch) { return (ch - 'A') < 26; }
  constexpr bool isspace(unsigned ch)
  {
    return ch == ' ' || (ch - '\t') < 5;
  }

  inline std::string to_string(double x)
  {
    std::string str = std::to_string(x);
    while (str.back() == '0')
      str.pop_back();
    if (str.back() == '.')
      str += '0';
    return str;
  }
  inline std::string to_string(int x)
  {
    return std::to_string(x);
  }
}

namespace lgf
{
  class resultCode : public bitCode<int8_t>
  {
  public:
    enum result : int8_t
    {
      default_result,
      success_result,
      failed_result
    };
    resultCode() : bitCode() { value = 0; }
    resultCode(int8_t v) : bitCode(int8_t(v)) {}
    static resultCode success()
    {
      return resultCode(int8_t(resultCode::result::success_result));
    }
    static resultCode fail()
    {
      return resultCode(int8_t(resultCode::result::failed_result));
    }

    static resultCode pass()
    {
      return resultCode(int8_t(resultCode::result::default_result));
    }

    bool isSuccess()
    {
      return check(success_result);
    }
  };
  class logicResult
  {
  public:
    logicResult() = delete;
    logicResult(logicResult &res) { value = res; }
    static logicResult success()
    {
      return logicResult(0);
    }
    static logicResult fail()
    {
      return logicResult(1);
    }
    bool getValue() const { return value; }
    bool operator==(const logicResult &a) { return value == a.getValue(); }
    operator bool() const { return value; }

  private:
    logicResult(bool t) { value = t; }
    bool value = 0;
  };

  class debug_guard
  {
  public:
    debug_guard(std::string pos) : id(pos)
    {
      std::cout << "[debug log] Enter: " << id << "..." << std::endl;
    }
    ~debug_guard()
    {
      std::cout << "[debug log] Exit: " << id << ". " << std::endl;
    }
    std::string id;
  };

  class symbolID
  {
  public:
    symbolID() = default;
    symbolID(const symbolID &a) : id(a.id) {}
    symbolID(std::string id) : id(id) {}
    std::string getID() const
    {
      return id.value();
    }
    void set_value(std::string id)
    {
      this->id = id;
    }
    bool has_value() const
    {
      return id.has_value();
    }
    symbolID &operator=(std::string id)
    {
      this->id = id;
      return *this;
    }
    bool operator==(const symbolID &a) const
    {
      if (!id.has_value() || !a.id.has_value())
        return false;
      return id == a.id;
    }
    symbolID &operator=(const symbolID &a)
    {
      id = a.id;
      return *this;
    }
    std::string value() { return id.value(); }

  private:
    std::optional<std::string> id;
  };

#define DEBUG_LOG_GUARD \
  debug_guard(__FUNCTION__);
}
#endif