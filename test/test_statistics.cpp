#include "../statistics.hpp"

#include <gtest/gtest.h>

using namespace jwm;
using namespace std;


TEST(Pearson_correlation_coefficient, singleton)
{
    std::vector<double> x(1), y(x);
    double r = Pearson_correlation_coefficient_a(begin(x), end(x), begin(y));
}

TEST(Pearson_correlation_coefficient, MathWorksEx1)
{
    vector<double> x = {7, 33/7, 3, 5, 2},
                   y = {3, 5, 1, 7, 2};
    
    auto r = Pearson_correlation_coefficient_a(begin(x), end(x), begin(y));
    ASSERT_EQ(0.4514558056, r);
}
