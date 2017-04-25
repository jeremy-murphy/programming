#include <benchmark/benchmark.h>

#include "../find.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

using namespace std;

static void BM_find_unguarded(benchmark::State &s)
{
    vector<int> v(s.range(0));
    iota(begin(v), end(v), 0);
    
    while (s.KeepRunning())
    {
        benchmark::DoNotOptimize(jwm::find_unguarded(begin(v), v.back()));
    }
}


static void BM_find(benchmark::State &s)
{
    vector<int> v(s.range(0));
    iota(begin(v), end(v), 0);
    
    while (s.KeepRunning())
    {
        benchmark::DoNotOptimize(find(begin(v), end(v), v.back()));
    }
}

BENCHMARK(BM_find_unguarded)->Range(8, 8<<20);
BENCHMARK(BM_find)->Range(8, 8<<20);

static void BM_find_if_unguarded_bind(benchmark::State &s)
{
    vector<int> v(s.range(0));
    iota(begin(v), end(v), 0);
    int x = v.back();
    auto const pred = bind(equal_to<>(), placeholders::_1, x);
    
    while (s.KeepRunning())
    {
        benchmark::DoNotOptimize(jwm::find_if_unguarded(begin(v), pred));
    }
}


static void BM_find_if_unguarded_lambda(benchmark::State &s)
{
    vector<int> v(s.range(0));
    iota(begin(v), end(v), 0);
    int x = v.back();
    auto const pred = [&](auto const &y){ return x == y; };
    
    while (s.KeepRunning())
    {
        benchmark::DoNotOptimize(jwm::find_if_unguarded(begin(v), pred));
    }
}


BENCHMARK(BM_find_if_unguarded_bind)->Range(8, 8<<20);
BENCHMARK(BM_find_if_unguarded_lambda)->Range(8, 8<<20);

BENCHMARK_MAIN();
