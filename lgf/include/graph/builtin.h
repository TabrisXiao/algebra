#ifndef LGF_BUILTIN_H_
#define LGF_BUILTIN_H_

#include "operation.h"
#include "utils/logger.h"
#include "env.h"

using namespace lgf;

class int32Attr : public singleData<int32_t>
{
public:
    int32Attr(int number) : singleData<int32_t>("int32", number) {}
    static attribute get(int number)
    {
        return attribute::get<int32Attr>(number);
    }
    virtual sid_t to_sid() override
    {
        return std::to_string(get_data());
    }
};

class float32Attr : public singleData<float>
{
public:
    float32Attr(float number) : singleData<float>("float32", number) {}
    static attribute get(float number)
    {
        return attribute::get<float32Attr>(number);
    }
    virtual sid_t to_sid() override
    {
        return std::to_string(get_data());
    }
};

class stringAttr : public singleData<std::string>
{
public:
    stringAttr(std::string str) : singleData<std::string>("string", str) {}
    static attribute get(std::string str)
    {
        return attribute::get<stringAttr>(str);
    }
    virtual sid_t to_sid() override
    {
        return get_data();
    }
};

class stringDesc : public descBase
{
public:
    stringDesc() : descBase("stringDesc") {}
    static description get()
    {
        return description::get<stringDesc>();
    }
};

class moduleOp : public operation
{

public:
    moduleOp(region *r = nullptr) : operation("module", r)
    {
        add_region();
    }
    static std::unique_ptr<moduleOp> build(region *reg)
    {
        return std::move(std::make_unique<moduleOp>(reg));
    }

    virtual std::string represent() override
    {
        return get_name() + " " + get_region(0)->represent();
    }
    virtual std::unique_ptr<operation> copy() override
    {
        return clone<moduleOp>();
    }
};

class defineOp : public operation
{
public:
    defineOp(region *r = nullptr) : operation("define", r)
    {
    }
    static std::unique_ptr<defineOp> build(region *reg, description desc)
    {
        auto op = std::make_unique<defineOp>(reg);
        auto ob = op->create_object(desc);
        return std::move(op);
    }

    virtual std::string represent() override
    {
        return internal_rep();
    }
    virtual std::unique_ptr<operation> copy() override
    {
        return clone<defineOp>();
    }
};

class funcOp : public operation
{
public:
    funcOp(region *r = nullptr) : operation("func", r)
    {
        add_region();
    }
    static std::unique_ptr<funcOp> build(region *reg)
    {
        return std::move(std::make_unique<funcOp>(reg));
    }

    virtual std::string represent() override
    {
        return get_sid() + " " + get_name() + " " + get_region(0)->represent();
    }
    virtual std::unique_ptr<operation> copy() override
    {
        return clone<funcOp>();
    }
};

class addOp : public operation
{
public:
    addOp(region *r = nullptr) : operation("add", r)
    {
        add_region();
    }
    static std::unique_ptr<addOp> build(region *reg, object *input1, object *input2)
    {
        auto op = std::make_unique<addOp>(reg);
        op->create_object(input1->get_desc());
        op->register_input(input1);
        op->register_input(input2);
        return std::move(op);
    }
    virtual std::string represent() override
    {
        return output(0).represent() + " = " + input(0).get_sid() + " + " + input(1).get_sid();
    }
    virtual std::unique_ptr<operation> copy() override
    {
        return clone<addOp>();
    }
};

class subtractOp : public operation
{
public:
    subtractOp(region *r = nullptr) : operation("subtract", r)
    {
        add_region();
    }
    static std::unique_ptr<subtractOp> build(region *reg, object *input1, object *input2)
    {
        auto op = std::make_unique<subtractOp>(reg);
        op->create_object(input1->get_desc());
        op->register_input(input1);
        op->register_input(input2);
        return std::move(op);
    }
    virtual std::string represent() override
    {
        return output(0).represent() + " = " + input(0).get_sid() + " - " + input(1).get_sid();
    }
    virtual std::unique_ptr<operation> copy() override
    {
        return clone<subtractOp>();
    }
};

class multiplyOp : public operation
{
public:
    multiplyOp(region *r = nullptr) : operation("multiply", r)
    {
        add_region();
    }
    static std::unique_ptr<multiplyOp> build(region *reg, object *input1, object *input2)
    {
        auto op = std::make_unique<multiplyOp>(reg);
        op->create_object(input1->get_desc());
        op->register_input(input1);
        op->register_input(input2);
        return std::move(op);
    }
    virtual std::string represent() override
    {
        return output(0).represent() + " = " + input(0).get_sid() + " * " + input(1).get_sid();
    }
    virtual std::unique_ptr<operation> copy() override
    {
        return clone<multiplyOp>();
    }
};

class divideOp : public operation
{
public:
    divideOp(region *r = nullptr) : operation("divide", r)
    {
        add_region();
    }
    static std::unique_ptr<divideOp> build(region *reg, object *input1, object *input2)
    {
        auto op = std::make_unique<divideOp>(reg);
        op->create_object(input1->get_desc());
        op->register_input(input1);
        op->register_input(input2);
        return std::move(op);
    }
    virtual std::string represent() override
    {
        return output(0).represent() + " = " + input(0).get_sid() + " / " + input(1).get_sid();
    }

    virtual std::unique_ptr<operation> copy() override
    {
        return clone<divideOp>();
    }
};

#endif