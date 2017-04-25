#include "../statistics.hpp"

#include <gtest/gtest.h>

using namespace jwm;
using namespace std;

TEST(Pearson_correlation_coefficient, singleton)
{
    std::vector<double> x(1), y(x);
    double r = Pearson_correlation_coefficient_a(begin(x), end(x), begin(y));
}
