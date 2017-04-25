#ifndef JWM_FUNCTIONAL_HPP
#define JWM_FUNCTIONAL_HPP

#include <utility>

namespace jwm
{

struct identity
{
    template <typename T>
    auto operator()(T &&x) const -> decltype(auto) 
    {
        return std::forward<T>(x);
    }
};

}

#endif
