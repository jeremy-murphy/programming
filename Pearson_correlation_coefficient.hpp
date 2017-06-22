#ifndef JWM_STATISTICS
#define JWM_STATISTICS

#include "functional.hpp"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/sum_kahan.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#include <boost/iterator/transform_iterator.hpp>

#include <Eigen/Dense>


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
    /**
     * BM_Pearson_correlation/8               79 ns         79 ns    8597511
     * BM_Pearson_correlation/64             780 ns        780 ns     878902
     * BM_Pearson_correlation/512           6566 ns       6537 ns     108931
     * BM_Pearson_correlation/4096         51484 ns      51480 ns      13549
     * BM_Pearson_correlation/32768       411962 ns     411976 ns       1689
     * BM_Pearson_correlation/262144     3453352 ns    3453108 ns        202
     * BM_Pearson_correlation/2097152   27945913 ns   27923286 ns         25
     * BM_Pearson_correlation/8388608  111367122 ns  111167390 ns          6
     */
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_concrete(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using std::sqrt;

        T xx{0}, xy{0}, yy{0};
        for (; x1 != xn; x1++, y1++)
        {
            xx += (*x1 - mean_x) * (*x1 - mean_x);
            xy += (*x1 - mean_x) * (*y1 - mean_y);
            yy += (*y1 - mean_y) * (*y1 - mean_y);
        }
        return xy / sqrt(xx * yy);
    }

    /**
     * BM_Pearson_correlation/8               22 ns         22 ns   31160632
     * BM_Pearson_correlation/64             110 ns        110 ns    6334243
     * BM_Pearson_correlation/512            810 ns        809 ns     863173
     * BM_Pearson_correlation/4096          6369 ns       6369 ns     109323
     * BM_Pearson_correlation/32768        51416 ns      51327 ns      13529
     * BM_Pearson_correlation/262144      484997 ns     485007 ns       1424
     * BM_Pearson_correlation/2097152    3976323 ns    3975979 ns        174
     * BM_Pearson_correlation/8388608   15962961 ns   15959515 ns         44
     */
    template <typename I, typename J>
    auto Pearson_correlation_coefficient_concrete_one_pass_naive(I x1, I xn, J y1)
    {
        assert(x1 != xn);
        using std::sqrt;
        
        using T = typename std::iterator_traits<I>::value_type;
        
        auto const n = std::distance(x1, xn);
        T mean_x{0}, mean_y{0}, xy{0}, xx{0}, yy{0};
        for (; x1 != xn; x1++, y1++)
        {
            xy += *x1 * *y1;
            xx += *x1 * *x1;
            yy += *y1 * *y1;
            mean_x += *x1;
            mean_y += *y1;
        }
        mean_x /= n;
        mean_y /= n;
        auto const Pcc = (xy - n * mean_x * mean_y) / sqrt((xx - n * mean_x * mean_x) * 
        (yy - n * mean_y * mean_y));
        return std::make_tuple(Pcc, mean_x, mean_y);
    }
    
    /**
     * BM_Pearson_correlation/8              134 ns        126 ns    5740335
     * BM_Pearson_correlation/64             583 ns        583 ns    1093192
     * BM_Pearson_correlation/512           4150 ns       4150 ns     164169
     * BM_Pearson_correlation/4096         32987 ns      32988 ns      20985
     * BM_Pearson_correlation/32768       265143 ns     265154 ns       2540
     * BM_Pearson_correlation/262144     2220244 ns    2220315 ns        302
     * BM_Pearson_correlation/2097152   19465385 ns   19466058 ns         37
     * BM_Pearson_correlation/8388608   76048524 ns   76051980 ns          9
     */
    template <typename I, typename J>
    auto Pearson_correlation_coefficient_concrete_one_pass_Kahan(I x1, I xn, J y1)
    {
        assert(x1 != xn);
        using std::sqrt;
        using namespace boost::accumulators;
        
        using T = typename std::iterator_traits<I>::value_type;
        
        auto const n = std::distance(x1, xn);
        accumulator_set<T, stats<tag::mean>> x, y;
        accumulator_set<T, stats<tag::sum_kahan>> xy, xx, yy;
        for (; x1 != xn; x1++, y1++)
        {
            xy(*x1 * *y1);
            xx(*x1 * *x1);
            yy(*y1 * *y1);
            x(*x1);
            y(*y1);
        }
        auto const Pcc = (sum_kahan(xy) - n * mean(x) * mean(y)) / sqrt((sum_kahan(xx) - n * mean(x) * mean(x)) * (sum_kahan(yy) - n * mean(y) * mean(y)));
        return std::make_tuple(Pcc, mean(x), mean(y));
    }
    
    /*
     * BM_Pearson_correlation/8             1078 ns       1055 ns     627524
     * BM_Pearson_correlation/64            2609 ns       2560 ns     262386
     * BM_Pearson_correlation/512          14256 ns      14009 ns      46700
     * BM_Pearson_correlation/4096        108848 ns     107231 ns       6542
     * BM_Pearson_correlation/32768       847712 ns     844315 ns        827
     * BM_Pearson_correlation/262144     6778388 ns    6713787 ns        102
     * BM_Pearson_correlation/2097152   52658159 ns   52426379 ns         13
     * BM_Pearson_correlation/8388608  209469991 ns  201800136 ns          3
     */
    //  2 x slower than the concrete base case!
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_concrete_p(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using std::sqrt;
        
        T xx{0}, xy{0}, yy{0};
        auto const n = xn - x1;
#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < n; ++i)
        {
            xx += (*x1 - mean_x) * (*x1 - mean_x);
            xy += (*x1 - mean_x) * (*y1 - mean_y);
            yy += (*y1 - mean_y) * (*y1 - mean_y);
            x1++, y1++;
        }
        return xy / sqrt(xx * yy);
    }
    
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
        
        transform(x1, xn, x1, bind2nd(minus<>(), mean_x));
        transform(y1, yn, y1, bind2nd(minus<>(), mean_y));
        
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
        
        auto fx1 = boost::make_transform_iterator(x1, bind2nd(std::minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, bind2nd(std::minus<>(), mean_y)),
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
        
        auto fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
             fxn = fx1 + n;
        auto fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y)),
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
        
        auto const fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
                   fxn = fx1 + distance(x1, xn);
        auto const fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        T numer, x_ss, y_ss;
        std::tie(numer, x_ss, y_ss) = three_way_inner_product(fx1, fxn, fy1, T(0), T(0), T(0));
        return numer / sqrt(x_ss * y_ss);
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
    
    
    template <typename BinaryOperator>
    vector_accumulation<BinaryOperator> make_vector_accumulation(BinaryOperator &&op)
    {
        return vector_accumulation<BinaryOperator>(std::forward<BinaryOperator>(op));
    }
    
    
    // (T, T) -> (T, T, T)
    // (x, y) -> (x*x, x*y, y*y)
    template <typename T, class Result=std::array<T, 3>>
    struct three_way_product
    {
        Result operator()(T x, T y) const
        {
            return {x * x, x * y, y * y};
        }
    };
    
    /*
     * BM_Pearson_correlation/8               86 ns         85 ns    8067122
     * BM_Pearson_correlation/64             835 ns        833 ns     839226
     * BM_Pearson_correlation/512           6871 ns       6859 ns     102247
     * BM_Pearson_correlation/4096         55074 ns      55076 ns      12683
     * BM_Pearson_correlation/32768       415214 ns     415226 ns       1624
     * BM_Pearson_correlation/262144     3473169 ns    3473258 ns        202
     * BM_Pearson_correlation/2097152   28190417 ns   28187235 ns         25
     * BM_Pearson_correlation/8388608  111119223 ns  111121497 ns          6
     */
    // This is the baseline algorithm.
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_d(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
                   fxn = fx1 + distance(x1, xn);
        auto const fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        auto const three_way = std::inner_product(fx1, fxn, fy1, std::array<T, 3>{}, make_vector_accumulation(std::plus<>()), three_way_product<T>());
        return three_way[1] / sqrt(three_way[0] * three_way[2]);
    }

    /* 
     * BM_Pearson_correlation/8               78 ns         78 ns    8723474
     * BM_Pearson_correlation/64             782 ns        782 ns     895837
     * BM_Pearson_correlation/512           6406 ns       6406 ns     109118
     * BM_Pearson_correlation/4096         51805 ns      51556 ns      13506
     * BM_Pearson_correlation/32768       413248 ns     412728 ns       1699
     * BM_Pearson_correlation/262144     3466678 ns    3457536 ns        203
     * BM_Pearson_correlation/2097152   28474369 ns   28437681 ns         25
     * BM_Pearson_correlation/8388608  112729810 ns  112651093 ns          6
     */
    template <typename I, typename J, typename T, typename Vector3=Eigen::Matrix<T, 3, 1>>
    auto Pearson_correlation_coefficient_eigen(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
                   fxn = fx1 + distance(x1, xn);
        auto const fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        auto const three_way = std::inner_product(fx1, fxn, fy1, Vector3(0, 0, 0), std::plus<>(), three_way_product<T, Vector3>());
        return three_way[1] / sqrt(three_way[0] * three_way[2]);
    }

    
    struct x_of_y
    {
        template <typename Functor, typename T>
        auto operator()(Functor &&x, T &&y) const
        {
            x(std::forward<T>(y));
            return std::forward<Functor>(x);
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
        auto const three_way = std::inner_product(fx1, fxn, fy1, std::array<kahan_accumulator, 3>{}, vector_accumulation<x_of_y>(x_of_y()), three_way_product<T>());
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
        auto fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
             fxn = fx1 + distance(x1, xn);
        auto fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        using namespace boost::accumulators;
        using kahan_accumulator = accumulator_set<T, stats<tag::sum_kahan>>;
        std::array<kahan_accumulator, 3> three_way;
        three_way_product<T> op2;
        for (; fx1 != fxn; fx1++, fy1++)
        {
            auto const tmp = op2(*fx1, *fy1);
            for_each(begin(three_way), end(three_way), begin(tmp), x_of_y());
        }
        auto const denom = sqrt(sum_kahan(three_way[0]) * sum_kahan(three_way[2]));
        return sum_kahan(three_way[1]) / denom;
    }
    

    // "foo" because I'm not sure what to call it yet.
    template <typename AccumulationContainer>
    struct foo
    {
        AccumulationContainer result;
        three_way_product<typename AccumulationContainer::value_type> op;
        
        template <typename T>
        foo(T &&result) : result{std::forward<T>(result)} {}
        
        template <typename T, typename U>
        void operator()(T x, U y)
        {
            auto const tmp = op(x, y);
            for_each(std::begin(result), std::end(result), std::begin(tmp), x_of_y());
        }
    };
    
    
    template <typename AccumulationContainer>
    foo<AccumulationContainer> make_foo(AccumulationContainer &&result)
    {
        return foo<AccumulationContainer>(std::forward<AccumulationContainer>(result));
    }
    
    /*
     * BM_Pearson_correlation/8              137 ns        137 ns    5101053
     * BM_Pearson_correlation/64            1226 ns       1226 ns     571494
     * BM_Pearson_correlation/512          10005 ns      10005 ns      71025
     * BM_Pearson_correlation/4096         78985 ns      78986 ns       8786
     * BM_Pearson_correlation/32768       634878 ns     634867 ns       1098
     * BM_Pearson_correlation/262144     5224304 ns    5224179 ns        132
     * BM_Pearson_correlation/2097152   41986322 ns   41985338 ns         17
     * BM_Pearson_correlation/8388608  169536467 ns  169538068 ns          4
     */
    // This looks to be equivalent in performance to f.
    template <typename I, typename J, typename T>
    auto Pearson_correlation_coefficient_g(I x1, I xn, J y1, T mean_x, T mean_y)
    {
        assert(x1 != xn);
        using namespace std;
        auto const fx1 = boost::make_transform_iterator(x1, bind2nd(minus<>(), mean_x)),
                   fxn = fx1 + distance(x1, xn);
        auto const fy1 = boost::make_transform_iterator(y1, bind2nd(minus<>(), mean_y));
        
        using namespace boost::accumulators;
        using kahan_accumulator = accumulator_set<T, stats<tag::sum_kahan>>;
        std::array<kahan_accumulator, 3> three_way;
        auto const result = for_each(fx1, fxn, fy1, make_foo(three_way)).result;
        auto const denom = sqrt(sum_kahan(result[0]) * sum_kahan(result[2]));
        return sum_kahan(result[1]) / denom;
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
        auto const Pcc = Pearson_correlation_coefficient_eigen(x1, xn, y1, mean_x, mean_y);
        return std::make_tuple(Pcc, mean_x, mean_y);
    }    
}

#endif
