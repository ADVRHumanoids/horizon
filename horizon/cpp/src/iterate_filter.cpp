#include "iterate_filter.h"
#include <limits>

IterateFilter::Pair::Pair():
    f(std::numeric_limits<double>::max()),
    h(std::numeric_limits<double>::max())
{}

bool IterateFilter::Pair::dominates(const IterateFilter::Pair &other, double beta, double gamma) const
{
    return f < other.f + gamma*other.h && beta*h < other.h;
}

bool IterateFilter::is_acceptable(const IterateFilter::Pair &test_pair) const
{
    auto dominates_test_pair = [&test_pair, this](const Pair& filt_pair)
    {
        return filt_pair.dominates(test_pair, beta, gamma);
    };

    return std::none_of(_entries.begin(),
                        _entries.end(),
                        dominates_test_pair);
}

bool IterateFilter::add(const IterateFilter::Pair &new_pair)
{
    for(auto& pair : _entries)
    {
        if(pair.dominates(new_pair))
        {
            return false;
        }
    }

    if(!is_acceptable(new_pair))
    {
        return false;
    }

    auto dominated_by_new_pair = [&new_pair](const Pair& filt_pair)
    {
        return new_pair.dominates(filt_pair);
    };

    _entries.remove_if(dominated_by_new_pair);

    auto new_pair_copy = new_pair;
    new_pair_copy.h = std::max(constr_tol, new_pair.h);

    _entries.push_back(new_pair_copy);

    return true;
}

void IterateFilter::clear()
{
    _entries.clear();
}

void IterateFilter::print()
{
    for(const auto& ent : _entries)
    {
        printf("(%4.3e, %4.3e) \n", ent.f, ent.h);
    }
}
