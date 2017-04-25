#include "../functional.hpp"

#include <gtest/gtest.h>

using namespace jwm;

identity const id;

TEST(identity, lvalue)
{
    int x = 1;
    auto &y = id(x);
    ASSERT_EQ(x, y);
    EXPECT_EQ(&x, &y);
}


TEST(identity, const_lvalue)
{
    int const x = 1;
    auto &y = id(x);
    ASSERT_EQ(x, y);
    EXPECT_EQ(&x, &y);
}


TEST(identity, rvalue)
{
    int x = 1;
    auto const &y = id(std::move(x));
    ASSERT_EQ(x, y);
}
