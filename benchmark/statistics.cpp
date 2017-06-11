#include <benchmark/benchmark.h>

#include "../Pearson_correlation_coefficient.hpp"

static void BM_Pearson_correlation(benchmark::State &s)
{
    std::vector<double> x(s.range(0)), y(s.range(0));

    while (s.KeepRunning())
    {
        benchmark::DoNotOptimize(jwm::Pearson_correlation_coefficient(begin(x), end(x), begin(y)));
    }
}

BENCHMARK(BM_Pearson_correlation)->Range(8, 8<<20);

BENCHMARK_MAIN();
