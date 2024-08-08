#ifndef ITERATEFILTER_H
#define ITERATEFILTER_H

#include <list>
#include <algorithm>
#include <iostream>
#include <limits>
#include "typedefs.h"

class IterateFilter
{

using Real=horizon::Real;

public:

    struct Pair
    {
        Real f;
        Real h;

        Pair();

        bool dominates(const Pair& other,
                       Real beta = 1.0,
                       Real gamma = 0.0) const;
    };

    IterateFilter() = default;

    bool is_acceptable(const Pair& test_pair) const;

    bool add(const Pair& new_pair);

    void clear();

    void print();

    Real beta = 1.0;
    Real gamma = 0.0;
    Real constr_tol = 1e-6;


private:

    std::list<Pair> _entries;

};

#endif // ITERATEFILTER_H
