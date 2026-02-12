#include "bounds.h"
using namespace horizon;

Bounds::Bounds(int dim, int n_nodes):
_dim(dim),
_n_nodes(n_nodes)
{
    _lower_bounds.resize(_dim, _n_nodes);
    _upper_bounds.resize(_dim, _n_nodes);
}

bool Bounds::setLowerBounds(const Eigen::MatrixXd& lower_bounds, const std::vector<int>& nodes)
{

    if (nodes.empty())
    {
        _lower_bounds = lower_bounds;
//        std::cout << "lb: \n" << lower_bounds << std::endl;
    }
    else
    {
//        std::cout << "lb at nodes: " << std::endl;

//        for (int elem : nodes)
//        {
//            std::cout << elem << " ";
//        }
//        std::cout << std::endl;

//        std::cout << "with values: \n" << lower_bounds << std::endl;

         for (int i = 0; i < nodes.size(); i++) {
             _lower_bounds(nodes[i]) = lower_bounds(i);
         }

    }


     return true;
}

bool Bounds::setUpperBounds(const Eigen::MatrixXd& upper_bounds, const std::vector<int>& nodes)
{
    if (nodes.empty())
    {
        _upper_bounds = upper_bounds;
//        std::cout << "ub: \n" << upper_bounds << std::endl;
    }
    else
    {
//        std::cout << "ub at nodes: " << std::endl;

//        for (int elem : nodes)
//        {
//            std::cout << elem << " ";
//        }
//        std::cout << std::endl;

//        std::cout << "with values: \n" << upper_bounds << std::endl;

         for (int i = 0; i < nodes.size(); i++) {
             _upper_bounds(nodes[i]) = upper_bounds(i);
         }

    }


     return true;
}

bool Bounds::setBounds(const Eigen::MatrixXd& lower_bounds, const Eigen::MatrixXd& upper_bounds, const std::vector<int>& nodes)
{
    setLowerBounds(lower_bounds, nodes);
    setUpperBounds(upper_bounds, nodes);
    return true;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Bounds::getBounds()
{
    return {_lower_bounds, _upper_bounds};
}

Eigen::MatrixXd Bounds::getLowerBounds()
{
    return _lower_bounds;
}

Eigen::MatrixXd Bounds::getUpperBounds()
{
    return _upper_bounds;
}
