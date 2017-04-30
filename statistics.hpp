#ifndef JWM_STATISTICS
#define JWM_STATISTICS

#include "functional.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum_kahan.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <numeric>

namespace jwm
{
    template <typename I, typename J>
    auto inner_product_nonempty(I f0, I l0, J f1)
    {
        // return std::inner_product(std::next(f0), l0, std::next(f1), *f0 * *f1);
        using namespace boost::accumulators;
        using T = typename std::iterator_traits<I>::value_type;
        accumulator_set<T, stats<tag::sum_kahan>> acc;
        for (; f0 != l0; ++f0, ++f1)
            acc(*f0 * *f1);
        return sum_kahan(acc);
    }
    
    
    // I'm using x1, xn, etc just to mimic the stats lingo.
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_a(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const yn = y1 + distance(x1, xn);
        
        transform(x1, xn, x1, bind2nd(minus<>(), mean_x));
        transform(y1, yn, y1, bind2nd(minus<>(), mean_y));
        
        auto const numer = inner_product_nonempty(x1, xn, y1);
        auto const x_ss = inner_product_nonempty(x1, xn, x1),
                   y_ss = inner_product_nonempty(y1, yn, y1);
        auto const denom = sqrt(x_ss * y_ss);
        return numer / denom;
    }

    
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_b(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);
        
        auto fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y)),
             fyn = fy1 + n;

        auto const numer = inner_product_nonempty(fx1, fxn, fy1);
        auto const x_ss = inner_product_nonempty(fx1, fxn, fx1),
                   y_ss = inner_product_nonempty(fy1, fyn, fy1);
        auto const denom = sqrt(x_ss * y_ss);
        return numer / denom;
    }

    
    template <typename I, typename J>
    auto Pearson_correlation_coefficient(I x1, I xn, J y1)
    {
        assert(x1 != xn);
        using T = typename std::iterator_traits<I>::value_type;
        auto const n = std::distance(x1, xn);
        auto const f = [&](I f, I l)
        {
            using namespace boost::accumulators;
            accumulator_set<T, stats<tag::sum_kahan>> acc;
            std::for_each(f, l, std::ref(acc));
            return sum_kahan(acc) / n;
        }; // or Boost accumulator
        auto const mean_x = f(x1, xn), 
                   mean_y = f(y1, y1 + n);
        return Pearson_correlation_coefficient_b(x1, xn, y1, mean_x, mean_y);
    }    
}

#endif
