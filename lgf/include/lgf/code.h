
#ifndef LGF_CODE_H_
#define LGF_CODE_H_

#include <string>

namespace lgf
{
    using code_t = uint64_t;
    using csize_t = uint8_t;
    class byteCodeHandle;
    class byteCodeStorage
    {
    public:
        byteCodeStorage() = default;
        byteCodeStorage(code_t val, csize_t h_) : data(val), h(h_) {}
        code_t get_data() { return data; }
        csize_t get_handle() const { return h; }
        code_t crop_data(csize_t i, csize_t j)
        {
            // the crop range is [i+1, j]
            if (i > j || j > 64)
            {
                auto istr = std::to_string(i);
                auto jstr = std::to_string(j);
                throw std::runtime_error("byteCodeStorage::crop_data: invalid crop range: " + istr + " > " + jstr + " or " + jstr + " > 64!");
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

        byteCodeHandle init_code(csize_t c_size);

    private:
        code_t data = 0;
        csize_t h = 0;
    };

    class byteCodeHandle
    {
    public:
        byteCodeHandle() = default;
        byteCodeHandle(csize_t h, csize_t s, byteCodeStorage *d) : offset_(h), size_(s), ptr_(d) {}
        virtual ~byteCodeHandle() = default;
        csize_t offset() const { return offset_; }
        csize_t size() const { return size_; }
        byteCodeStorage *ptr() { return ptr_; }

        bool has(const code_t &val)
        {
            return ptr_->is_one_at(val + offset_);
        }
        bool is(const code_t &val)
        {
            return ptr_->is_equal(offset_, offset_ + size_ - 1, val);
        }
        void add(const code_t &val)
        {
            ptr_->add_data(offset_, offset_ + size_ - 1, 1ULL << val);
        }
        void write(const code_t &val)
        {
            ptr_->write_data(offset_, offset_ + size_ - 1, val);
        }
        void reset()
        {
            ptr_->write_data(offset_, offset_ + size_ - 1, 0);
        }

    private:
        csize_t offset_, size_;
        byteCodeStorage *ptr_;
    };

} // namespace lgf
#endif