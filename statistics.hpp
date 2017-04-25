#ifndef JWM_STATISTICS
#define JWM_STATISTICS

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <numeric>

namespace jwm
{
    // I'm using x1, xn, etc just to mimic the stats lingo.
    template <typename I, typename J>
    auto Pearson_correlation_coefficient_a(I x1, I xn, J y1) -> decltype(auto)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);
        auto const yn = y1 + n;
        auto x_mean = accumulate(next(x1), xn, *x1) / n,
             y_mean = accumulate(next(y1), yn, *y1) / n;
             
        transform(x1, xn, x1, bind(minus<>(), placeholders::_1, x_mean));
        transform(y1, yn, y1, bind(minus<>(), placeholders::_1, y_mean));
        
        auto const numer = inner_product(next(x1), xn, next(y1), *x1 * *y1);
        auto const x_ss = inner_product(next(x1), xn, next(x1), *x1 * *x1),
                   y_ss = inner_product(next(y1), yn, next(y1), *y1 * *y1);
        auto const denom = sqrt(x_ss) * sqrt(y_ss);
        assert(denom);
        return numer / denom;
    }
}

#endif
