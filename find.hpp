#ifndef JWM_FIND_HPP
#define JWM_FIND_HPP

#include <functional>
#include <iterator>

namespace jwm
{
    template <typename Iterator, typename Predicate>
    Iterator find_if_unguarded(Iterator first, Predicate pred)
    {
        while (!pred(*first))
            ++first;
        return first;
    }
    
    
    template <typename Iterator, typename T>
    Iterator find_unguarded(Iterator first, T &&x)
    {
        using namespace std;
        auto const pred = bind(equal_to<>(), placeholders::_1, forward<T>(x));
        return find_if_unguarded(first, pred);
    }
}

#endif
