#include "functions.h"

using namespace horizon;

Function::Function(casadi::Function fun, int n_nodes):
    _n_nodes(n_nodes)
{
    _name = fun.name();
    _dim = fun.size1_out(0);
    _fun = fun;

//    _nodes.resize(n_nodes);
//    for (int i = 0; i < n_nodes; ++i)
//    {
//        _nodes[i] = i;
//    }

}

std::vector<int> Function::getNodes()
{
    return _nodes;
}

std::string Function::getName()
{
    return _name;
}

int Function::getDim()
{
    return _dim;
}

casadi::Function Function::getFunction()
{
    return _fun;
}

Constraint::Constraint(casadi::Function fun, int n_nodes):
    Function(fun, n_nodes)
{
    _bounds = std::make_unique<Bounds>(getDim(), n_nodes);
}

bool Constraint::setNodes(std::vector<int> nodes, bool erasing)
{
    // also setting nodes to _bounds?
//    std::cout << "erasing: " << erasing << std::endl;
//    std::cout << "nodes: " << std::endl;
//    for (auto node : nodes)
//    {
//        std::cout << node << " ";
//    }
//    std::cout << std::endl;

    if (erasing) {
        _nodes = std::move(nodes);
    } else {
        _nodes.insert(_nodes.end(), nodes.begin(), nodes.end());
        std::sort(_nodes.begin(), _nodes.end());
        _nodes.erase(std::unique(_nodes.begin(), _nodes.end()), _nodes.end());
    }
    return true;
}

bool Constraint::setLowerBounds(Eigen::MatrixXd lb, std::vector<int> nodes)
{
    return _bounds->setLowerBounds(lb, nodes);
}

bool Constraint::setUpperBounds(Eigen::MatrixXd ub, std::vector<int> nodes)
{
    return _bounds->setUpperBounds(ub, nodes);
}

bool Constraint::setBounds(Eigen::MatrixXd lb, Eigen::MatrixXd ub, std::vector<int> nodes)
{
    return _bounds->setBounds(lb, ub, nodes);
}

Eigen::MatrixXd Constraint::getLowerBounds()
{
    return _bounds->getLowerBounds();
}

Eigen::MatrixXd Constraint::getUpperBounds()
{
    return _bounds->getUpperBounds();
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Constraint::getBounds()
{
    return _bounds->getBounds();
}

Cost::Cost(casadi::Function fun, int n_nodes):
    Function(fun, n_nodes)
{
}

bool Cost::setNodes(std::vector<int> nodes, bool erasing)
{

//    std::cout << "erasing: " << erasing << std::endl;
//    std::cout << "nodes: " << std::endl;
//    for (auto node : nodes)
//    {
//        std::cout << node << " ";
//    }
//    std::cout << std::endl;

    if (erasing) {
        _nodes = std::move(nodes);
    } else {
        _nodes.insert(_nodes.end(), nodes.begin(), nodes.end());
        std::sort(_nodes.begin(), _nodes.end());
        _nodes.erase(std::unique(_nodes.begin(), _nodes.end()), _nodes.end());
    }
    return true;
}

Residual::Residual(casadi::Function fun, int n_nodes):
    Function(fun, n_nodes)
{
}

bool Residual::setNodes(std::vector<int> nodes, bool erasing)
{

//    std::cout << "erasing: " << erasing << std::endl;
//    std::cout << "nodes: " << std::endl;
//    for (auto node : nodes)
//    {
//        std::cout << node << " ";
//    }
//    std::cout << std::endl;

    if (erasing) {
        _nodes = std::move(nodes);
    } else {
        _nodes.insert(_nodes.end(), nodes.begin(), nodes.end());
        std::sort(_nodes.begin(), _nodes.end());
        _nodes.erase(std::unique(_nodes.begin(), _nodes.end()), _nodes.end());
    }
    return true;
}

