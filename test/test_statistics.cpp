#include <hpx/hpx_main.hpp>

#include "../statistics.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace jwm;
using namespace std;

namespace {

TEST(Pearson_correlation_coefficient, NaN)
{
    std::vector<double> x(1), y(x); // zero
    double r = Pearson_correlation_coefficient(begin(x), end(x), begin(y));
    ASSERT_TRUE(std::isnan(r));
    x[0] = y[0] = 1.0;
    r = Pearson_correlation_coefficient(begin(x), end(x), begin(y));
    ASSERT_TRUE(std::isnan(r));
    x[0] = y[0] = -1.0;
    r = Pearson_correlation_coefficient(begin(x), end(x), begin(y));
    ASSERT_TRUE(std::isnan(r));
}

/*
TEST(Pearson_correlation_coefficient, MathWorksEx1)
{
    vector<double> x = {7, 33/7, 3, 5, 2},
                   y = {3, 5, 1, 7, 2};
    
    auto r = Pearson_correlation_coefficient(begin(x), end(x), begin(y));
    ASSERT_FLOAT_EQ(0.4514558056, r);
}
*/

TEST(Pearson_correlation_coefficient, StatisticsHowTo)
{
    vector<double> const x = {43, 21, 25, 42, 57, 59},
                         y = {99, 65, 79, 75, 87, 81};
    
    auto r = Pearson_correlation_coefficient(begin(x), end(x), begin(y));
    ASSERT_FLOAT_EQ(0.529809, r);
}

}


int hpx_main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto result = RUN_ALL_TESTS();
    return hpx::finalize(result);
}

int main(int argc, char **argv)
{
    return hpx::init(argc, argv);
}
