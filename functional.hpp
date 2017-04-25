#ifndef JWM_FUNCTIONAL_HPP
#define JWM_FUNCTIONAL_HPP

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

}

#endif
