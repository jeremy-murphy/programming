#ifndef JWM_STATISTICS
#define JWM_STATISTICS

#include "functional.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum_kahan.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <numeric>
#include <tuple>

namespace jwm
{
    template <typename I>
    auto Euclidean_norm(I first, I last)
    {
        using T = typename std::iterator_traits<I>::value_type;
        
        if (first == last)
            return T{0};
        
        using std::sqrt;
        return sqrt(std::inner_product(std::next(first), last, *first));
    }

    
    template <typename Iterator0, typename Iterator1>
    auto inner_product_nonempty(Iterator0 f0, Iterator0 l0, Iterator1 f1)
    {
        // return std::inner_product(std::next(f0), l0, std::next(f1), *f0 * *f1);
        using namespace boost::accumulators;
        using T = typename std::iterator_traits<Iterator0>::value_type;
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
        
        transform(x1, xn, x1, std::bind2nd(minus<>(), mean_x));
        transform(y1, yn, y1, std::bind2nd(minus<>(), mean_y));
        
        auto const numer = inner_product_nonempty(x1, xn, y1);
        auto const x_ss = inner_product_nonempty(x1, xn, x1),
                   y_ss = inner_product_nonempty(y1, yn, y1);
        auto const denom = sqrt(x_ss * y_ss);
        return numer / denom;
    }

    
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_ba(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using std::sqrt;
        auto const n = std::distance(x1, xn);
        
        auto fx1 = boost::make_transform_iterator(x1, std::bind2nd(std::minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, std::bind2nd(std::minus<>(), mean_y)),
             fyn = fy1 + n;

        auto const numer = inner_product_nonempty(fx1, fxn, fy1);
        auto const x_ss = inner_product_nonempty(fx1, fxn, fx1),
                   y_ss = inner_product_nonempty(fy1, fyn, fy1);
        auto const denom = sqrt(x_ss * y_ss);
        return numer / denom;
    }

    
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_bb(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);
        
        auto fx1 = boost::make_transform_iterator(x1, std::bind2nd(minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, std::bind2nd(minus<>(), mean_y)),
             fyn = fy1 + n;
        
        auto const numer = inner_product_nonempty(fx1, fxn, fy1);
        auto const x_ss = Euclidean_norm(fx1, fxn),
                   y_ss = Euclidean_norm(fy1, fyn);
        auto const denom = x_ss * y_ss;
        return numer / denom;
    }
    
    
    template <typename I, typename J, typename T, typename CommutativeBinaryOperator, typename AssociativeBinaryOperator>
    std::tuple<T, T, T> three_way_inner_product(I f0, I l0, J f1, T a, T b, T c, CommutativeBinaryOperator op1, AssociativeBinaryOperator op2)
    {
        for (; f0 != l0; ++f0, ++f1)
        {
            a = op1(a, op2(*f0, *f1));
            b = op1(b, op2(*f0, *f0));
            c = op1(c, op2(*f1, *f1));
        }
        return std::make_tuple(a, b, c);
    }

    
    template <typename I, typename J, typename T>
    std::tuple<T, T, T> three_way_inner_product(I f0, I l0, J f1, T a, T b, T c)
    {
        return three_way_inner_product(f0, l0, f1, a, b, c, std::plus<>(), std::multiplies<>());
    }
    
    
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_c(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);
        
        auto fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        T numer, x_ss, y_ss;
        std::tie(numer, x_ss, y_ss) = three_way_inner_product(fx1, fxn, fy1, T(0), T(0), T(0));
        auto const denom = sqrt(x_ss * y_ss);
        return numer / denom;
    }
    
    
    // Vector addition, in the mathematical sense.
    template <typename BinaryOperator>
    struct vector_accumulation
    {
        BinaryOperator op;
        
        vector_accumulation(BinaryOperator op) : op{op} {}
        
        template <typename U, typename V>
        auto operator()(U &&x, V &&y) const
        {
            assert(x.size() <= y.size());
            using namespace std;
            typename remove_reference<typename remove_cv<U>::type>::type result;
            transform(begin(forward<U>(x)), end(forward<U>(x)), begin(forward<V>(y)), begin(result), op);
            return result;
        }
    };
    
    
    // (x, y) -> (x*x, x*y, y*y)
    struct three_way_product
    {
        template <typename T>
        std::array<T, 3> operator()(T x, T y) const
        {
            return {{x * x, x * y, y * y}};
        }
    };
    
    /*
     * BM_Pearson_correlation/8               87 ns         87 ns    7448864
     * BM_Pearson_correlation/64             864 ns        863 ns     841536
     * BM_Pearson_correlation/512           6985 ns       6985 ns     100748
     * BM_Pearson_correlation/4096         55691 ns      55672 ns      12673
     * BM_Pearson_correlation/32768       412155 ns     412132 ns       1683
     * BM_Pearson_correlation/262144     3635071 ns    3635034 ns        201
     * BM_Pearson_correlation/2097152   29817785 ns   29815147 ns         25
     * BM_Pearson_correlation/8388608  110969886 ns  110971045 ns          6
     */
    // This is the baseline algorithm.
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_d(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);        
        auto const fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
                   fxn = fx1 + n;
        auto const fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));

        auto const three_way = std::inner_product(fx1, fxn, fy1, std::array<T, 3>{}, vector_accumulation<std::plus<>>(std::plus<>()), three_way_product());
        auto const denom = sqrt(three_way[0] * three_way[2]);
        return three_way[1] / denom;
    }
    
    
    struct x_of_y
    {
        template <typename Functor, typename U>
        auto operator()(Functor &&x, U &&y) const
        {
            x(std::forward<U>(y));
            return x;
        }
    };
    
    
    /*
     * BM_Pearson_correlation/8              213 ns        213 ns    3289895
     * BM_Pearson_correlation/64            1833 ns       1831 ns     383951
     * BM_Pearson_correlation/512          14873 ns      14873 ns      47319
     * BM_Pearson_correlation/4096        114038 ns     114042 ns       5791
     * BM_Pearson_correlation/32768       908694 ns     908651 ns        783
     * BM_Pearson_correlation/262144     7389450 ns    7389580 ns         94
     * BM_Pearson_correlation/2097152   59193065 ns   59175501 ns         12
     * BM_Pearson_correlation/8388608  234984649 ns  234989310 ns          3
     */
    // This variation is 100% slower than d.
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_e(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);        
        auto const fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
        fxn = fx1 + n;
        auto const fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        using namespace boost::accumulators;
        using kahan_accumulator = accumulator_set<T, stats<tag::sum_kahan>>;
        auto const three_way = std::inner_product(fx1, fxn, fy1, std::array<kahan_accumulator, 3>{}, vector_accumulation<x_of_y>(x_of_y()), three_way_product());
        auto const denom = sqrt(sum_kahan(three_way[0]) * sum_kahan(three_way[2]));
        return sum_kahan(three_way[1]) / denom;
    }
    
    
    template <typename I, typename J, typename F>
    F for_each(I first0, I last0, J first1, F &&f)
    {
        for ( ; first0 != last0; ++first0, ++first1)
            f(*first0, *first1);
        return std::move(f);
    }
    
    
    /*
     * BM_Pearson_correlation/8              133 ns        133 ns    5112730
     * BM_Pearson_correlation/64            1215 ns       1215 ns     577956
     * BM_Pearson_correlation/512           9776 ns       9772 ns      71737
     * BM_Pearson_correlation/4096         78133 ns      78098 ns       8726
     * BM_Pearson_correlation/32768       624688 ns     624383 ns       1110
     * BM_Pearson_correlation/262144     5299817 ns    5292250 ns        129
     * BM_Pearson_correlation/2097152   44251541 ns   44182891 ns         16
     * BM_Pearson_correlation/8388608  173088181 ns  172820258 ns          4
     */
    // This variation is only 50% slower than d.
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_f(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const n = distance(x1, xn);        
        auto fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        using namespace boost::accumulators;
        using kahan_accumulator = accumulator_set<T, stats<tag::sum_kahan>>;
        std::array<kahan_accumulator, 3> three_way;
        three_way_product op2;
        for (; fx1 != fxn; fx1++, fy1++)
        {
            auto const tmp = op2(*fx1, *fy1);
            for_each(begin(three_way), end(three_way), begin(tmp), x_of_y());
        }
        auto const denom = sqrt(sum_kahan(three_way[0]) * sum_kahan(three_way[2]));
        return sum_kahan(three_way[1]) / denom;
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
        return Pearson_correlation_coefficient_f(x1, xn, y1, mean_x, mean_y);
    }    
}

#endif
