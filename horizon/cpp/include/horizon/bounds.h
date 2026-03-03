#ifndef BOUNDS_H
#define BOUNDS_H

#include <Eigen/Dense>
#include <memory>
#include <iostream>

namespace horizon
{

class Bounds
{

public:

    typedef std::shared_ptr<Bounds> Ptr;

    Bounds(int dim, int n_nodes);
    bool setBounds(const Eigen::MatrixXd& lower_bounds, const Eigen::MatrixXd& upper_bounds, const std::vector<int>& nodes = {});
    bool setLowerBounds(const Eigen::MatrixXd& lower_bounds, const std::vector<int>& nodes = {});
    bool setUpperBounds(const Eigen::MatrixXd& upper_bounds, const std::vector<int>& nodes = {});

    Eigen::MatrixXd getLowerBounds();
    Eigen::MatrixXd getUpperBounds();
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getBounds();

private:

    int _dim;
    int _n_nodes;
    std::vector<int> _nodes;
    Eigen::MatrixXd _lower_bounds;
    Eigen::MatrixXd _upper_bounds;

};
}

#endif // BOUNDS_H
