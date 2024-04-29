#include "unit_test_frame.h"

namespace test_body
{
    using code_t = uint64_t;
    using csize_t = uint8_t;

    class code
    {
    public:
        code() = default;
        code_t get_data() { return data; }
        code_t crop_data(csize_t i, csize_t j)
        {
            // the crop range is [i+1, j]
            if (i > j || j > 64)
            {
                auto istr = std::to_string(i);
                auto jstr = std::to_string(j);
                throw std::runtime_error("code::crop_data: invalid crop range: " + istr + " > " + jstr + " or " + jstr + " > 64!");
            }
            code_t mask = j == 64 ? (~0ULL) << i : ((1ULL << (j - i + 1)) - 1) << i;
            return (data & mask) >> i;
        }
        void write_data(csize_t i, csize_t j, csize_t val)
        {
            code_t mask = ((1ULL << (j - i + 1)) - 1) << i;
            data = (data & ~mask) | (val << i);
        }
        void add_data(csize_t i, csize_t j, csize_t val)
        {
            code_t mask = ((1ULL << (j - i + 1)) - 1) << i;
            data = data | (val << i);
        }
        csize_t append_data(code_t val, csize_t size)
        {
            if (h + size > 64)
                return 64;
            write_data(h, h + size - 1, val);
            csize_t mark = h;
            h += size;
            return mark;
        }
        code_t data = 0;
        csize_t h = 0;
    };

    class decoder
    {
    public:
        decoder(csize_t h, csize_t s, code *d) : handle(h), size(s), ptr(d) {}
        void set_sid(sid_t id_) { id = id_; }
        sid_t get_sid() { return id; }

    private:
        sid_t id;
        const csize_t handle, size;
        const code *ptr;
    };

    class encoder
    {
    public:
        encoder() = default;
        decoder init_code(code &data, csize_t c_size)
        {
            auto h = data.append_data(0, c_size);
            if (h == 64)
                throw std::runtime_error("encoder::exclusive_encode: data overflow!");
            return decoder(h, c_size, &data);
        }
    };
}

class test_obj : public test_wrapper
{
public:
    test_obj() { test_id = "object test"; };
    bool run()
    {
        test_body::code c1, c2;
        auto n = c1.append_data(5, 3);
        c1.append_data(1, 2);
        c1.append_data(1, 2);
        //      01 01 101
        std::cout << c1.get_data() << std::endl;
        std::cout << c1.crop_data(n, n + 232) << std::endl;
        return 0;
    }
};