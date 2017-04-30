#ifndef JWM_FUNCTIONAL_HPP
#define JWM_FUNCTIONAL_HPP

#include <functional>
#include <utility>

namespace jwm
{

struct identity
{
    template <typename T>
    T &&operator()(T &&x) const
    {
        return std::forward<T>(x);
    }
};


template <typename F, typename T>
auto bind1st(F &&f, T &&x)
{
    return std::bind(std::forward<F>(f), std::forward<T>(x), std::placeholders::_1);
}


template <typename F, typename T>
auto bind2nd(F &&f, T &&x)
{
    return std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<T>(x));
}

}

#endif
