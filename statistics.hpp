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
    template <typename I, typename BinaryOperator=std::plus<>>
    auto accumulate_nonempty(I first, I last, BinaryOperator op=BinaryOperator()) -> decltype(auto)
    {
        return std::accumulate(std::next(first), last, *first, op);
    }

    
    template <typename I, typename J>
    auto inner_product_nonempty(I f0, I l0, J f1)
    {
        return std::inner_product(std::next(f0), l0, std::next(f1), *f0 * *f1);
    }
    
    
    // I'm using x1, xn, etc just to mimic the stats lingo.
    template <typename I, typename J, typename F>
    auto Pearson_correlation_coefficient_a(I x1, I xn, J y1, F mean) -> decltype(auto)
    {
        assert(x1 != xn);
        using namespace std;
        auto const yn = y1 + distance(x1, xn);
        
        transform(x1, xn, x1, bind(minus<>(), placeholders::_1, mean(x1, xn)));
        transform(y1, yn, y1, bind(minus<>(), placeholders::_1, mean(y1, yn)));
        
        auto const numer = inner_product_nonempty(x1, xn, y1);
        auto const x_ss = inner_product_nonempty(x1, xn, x1),
                   y_ss = inner_product_nonempty(y1, yn, y1);
        auto const denom = sqrt(x_ss) * sqrt(y_ss);
        assert(denom);
        return numer / denom;
    }

    
    template <typename I, typename J>
    auto Pearson_correlation_coefficient_a(I x1, I xn, J y1) -> decltype(auto)
    {
        auto const mean = [&](I f, I l){ return accumulate_nonempty(f, l) / std::distance(f, l); }; // or Boost accumulator
        return Pearson_correlation_coefficient_a(x1, xn, y1, mean);
    }
    
}

#endif
