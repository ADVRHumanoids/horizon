#ifndef ITERATEFILTER_H
#define ITERATEFILTER_H

#include <list>
#include <algorithm>
#include <iostream>

class IterateFilter
{

public:

    struct Pair
    {
        double f;
        double h;

        Pair();

        bool dominates(const Pair& other,
                       double beta = 1.0,
                       double gamma = 0.0) const;
    };

    IterateFilter() = default;

    bool is_acceptable(const Pair& test_pair) const;

    bool add(const Pair& new_pair);

    void clear();

    void print();

    double beta = 1.0;
    double gamma = 0.0;
    double constr_tol = 1e-6;


private:

    std::list<Pair> _entries;

};

#endif // ITERATEFILTER_H
