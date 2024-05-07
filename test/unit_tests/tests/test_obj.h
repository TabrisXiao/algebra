#include "unit_test_frame.h"
#include <map>
#include <string>

namespace test_body
{
    using code_t = uint64_t;
    using csize_t = uint8_t;
    using sid_t = std::string;
    class xBitCode;
    class bitCodeHandle
    {
    public:
        bitCodeHandle() = default;
        bitCodeHandle(csize_t h, csize_t s, xBitCode *d) : offset_(h), size_(s), ptr_(d) {}
        virtual ~bitCodeHandle() = default;
        csize_t offset() const { return offset_; }
        csize_t size() const { return size_; }
        xBitCode *ptr() { return ptr_; }

    private:
        csize_t offset_, size_;
        xBitCode *ptr_;
    };
    class xBitCode
    {
    public:
        xBitCode() = default;
        xBitCode(code_t val, csize_t h_) : data(val), h(h_) {}
        code_t get_data() { return data; }
        code_t crop_data(csize_t i, csize_t j)
        {
            // the crop range is [i+1, j]
            if (i > j || j > 64)
            {
                auto istr = std::to_string(i);
                auto jstr = std::to_string(j);
                throw std::runtime_error("xBitCode::crop_data: invalid crop range: " + istr + " > " + jstr + " or " + jstr + " > 64!");
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
        bool is_one_at(csize_t i)
        {
            return (data & (1ULL << i)) != 0;
        }
        bool is_equal(csize_t i, csize_t j, code_t val)
        {
            code_t mask = ((1ULL << (j - i + 1)) - 1) << i;
            return (data & mask) == val;
        }
        bool is_equal(code_t val)
        {
            return data == val;
        }
        void reset()
        {
            data = 0;
            h = 0;
        }

        bitCodeHandle init_code(csize_t c_size)
        {
            auto h = append_data(0, c_size);
            if (h == 64)
                throw std::runtime_error("xBitCode::init_code: data overflow!");
            return bitCodeHandle(h, c_size, this);
        }

    private:
        code_t data = 0;
        csize_t h = 0;
    };

    class translator
    {
    public:
        translator() = default;

    protected:
        bool has(const code_t &val, bitCodeHandle &h)
        {
            return h.ptr()->is_one_at(val + h.offset());
        }
        bool is(const code_t &val, bitCodeHandle &h)
        {
            return h.ptr()->is_equal(h.offset(), h.offset() + h.size() - 1, val);
        }
        void add(const code_t &val, bitCodeHandle &h)
        {
            h.ptr()->add_data(h.offset(), h.offset() + h.size() - 1, 1ULL << val);
        }
        void write(const code_t &val, bitCodeHandle &h)
        {
            h.ptr()->write_data(h.offset(), h.offset() + h.size() - 1, val);
        }
        void reset(bitCodeHandle &h)
        {
            // reset the xBitCode to 0
            h.ptr()->write_data(h.offset(), h.offset() + h.size() - 1, 0);
        }
    };

    class propertyBitCode : public translator
    {
    public:
        propertyBitCode(csize_t size, xBitCode *base) : h(base->init_code(size)) {}
        virtual ~propertyBitCode() {}
        bool has(code_t p)
        {
            return translator::has(p, h);
        }
        void add(code_t p)
        {
            translator::add(p, h);
        }
        bool is(code_t p)
        {
            return translator::is(p, h);
        }
        void write(code_t p)
        {
            translator::write(p, h);
        }

        virtual sid_t represent() = 0;

    public:
        bitCodeHandle h;
    };

    class algebraPropertyCode : public propertyBitCode
    {
    public:
        enum property : code_t
        {
            commutable, // A * B = B * A;
            identity,   // means p * p = p;
        };

        algebraPropertyCode(xBitCode *base) : propertyBitCode(30, base) {}

        virtual sid_t represent() override
        {
            sid_t res = "";
            if (has(commutable))
            {
                res += "commutable, ";
            }
            if (has(identity))
            {
                res += "identity, ";
            }
            if (res.size() > 0)
            {
                res.substr(0, res.size() - 2);
                res = "property: " + res;
            }
            return res;
        }
    };

    class specialNumberCode : public propertyBitCode
    {
    public:
        enum value
        {
            finite,
            zero,
            one,
            e,
            pi,
        };
        specialNumberCode(xBitCode *base) : propertyBitCode(4, base) {}
        virtual sid_t represent() override
        {
            sid_t res = "";
            if (is(zero))
            {
                res += "zero";
            }
            else if (is(one))
            {
                res += "one";
            }
            else if (is(e))
            {
                res += "e";
            }
            else if (is(pi))
            {
                res += "pi";
            }
            if (res.size() > 0)
            {
                res.substr(0, res.size() - 2);
                res = "value= " + res;
            }
            return res;
        }
    };

    class realNumberProperty : protected xBitCode
    {
    public:
        realNumberProperty() : value(this), property(this)
        {
            property.add(algebraPropertyCode::commutable);
        }
        sid_t represent()
        {
            sid_t res = value.represent();
            if (res.size() > 0)
                res += ", ";
            res += property.represent();
            return res;
        }
        code_t represent_value()
        {
            return get_data();
        }

    public:
        specialNumberCode value;
        algebraPropertyCode property;
    };

    template <typename T>
    class morphism_wrapper
    {
        // this object is used to wrapper all classes derived from base type T into a
        //
    public:
        morphism_wrapper() = default;

        morphism_wrapper(const morphism_wrapper &src)
        {
            ptr = std::move(src.get()->copy());
        }
        T *get() const { return ptr.get(); }
        template <typename U>
        U &dyn_cast()
        {
            auto p = dynamic_cast<U *>(ptr.get());
            return *p;
        }

    public:
        std::unique_ptr<T> ptr = nullptr;
    };

    class type
    {
    public:
        std::string id = "type";
        virtual std::unique_ptr<type> copy() { return std::make_unique<type>(); }
    };

    class real : public type
    {
    public:
        real() { id = "real"; }
        virtual std::unique_ptr<type> copy() override
        {
            return std::make_unique<real>();
        }
    };
} // namespace test_body
using namespace test_body;

class test_obj : public test_wrapper
{
public:
    test_obj() { test_id = "object test"; };
    bool run()
    {
        morphism_wrapper<type> a;
        a.ptr = std::make_unique<real>();
        morphism_wrapper<type> b(a);
        std::cout << "name: " << b.get()->id << std::endl;
        return 0;
    }
};