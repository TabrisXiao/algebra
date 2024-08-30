#ifndef AOC_CONVENTION_H
#define AOC_CONVENTION_H

namespace aoc
{

    class logicResult
    {
    public:
        logicResult() = default;
        logicResult(logicResult &res) { value = res; }
        ~logicResult() = default;
        static logicResult success()
        {
            return logicResult(0);
        }
        static logicResult fail()
        {
            return logicResult(1);
        }
        bool is_fail() { return value == 1; }
        bool is_success() { return value == 0; }
        bool getValue() const { return value; }
        bool operator==(const logicResult &a) { return value == a.getValue(); }
        operator bool() const { return value; }

    private:
        logicResult(bool t) { value = t; }
        bool value = 0;
    };
}

#endif