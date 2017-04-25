#include "../find.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <iosfwd>

using namespace jwm;
using namespace std;

template <typename Iterator>
ostream &operator<<(ostream &os, Iterator it)
{
    os << *it;
    return os;
}


vector<int> const data = {0, 1, 2, 3, 4};

TEST(find, predicate)
{
    auto const pred = bind(greater<>(), placeholders::_1, 2);
    auto const y = find_if_unguarded(begin(data), pred);
    ASSERT_EQ(begin(data) + 3, y);
}


TEST(find, rvalue)
{
    int x = 1;
    auto const y = find_unguarded(begin(data), move(x));
    ASSERT_EQ(find(begin(data), end(data), move(x)), y);
}


TEST(find, lvalue)
{
    int x = 1;
    auto const y = find_unguarded(begin(data), x);
    ASSERT_EQ(find(begin(data), end(data), x), y);
}


TEST(find, const_lvalue)
{
    int const x = 1;
    auto const y = find_unguarded(begin(data), x);
    ASSERT_EQ(find(begin(data), end(data), x), y);
}
