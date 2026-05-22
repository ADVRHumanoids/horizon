#include "horizon/variables.h"
#include "horizon/bounds.h"

using namespace horizon;


Variable::Variable(std::string name, int dim, int n_nodes):
    _name(name),
    _dim(dim),
    _n_nodes(n_nodes)
{

    _initial_guess.resize(_dim, _n_nodes);
    _bounds = std::make_unique<Bounds>(_dim, _n_nodes);

    _sym = casadi::SX::sym(_name, _dim);
}

bool Variable::setNodes(std::vector<int> nodes, bool erasing)
{
    _nodes = nodes;
    return true;
}

bool Variable::setLowerBounds(Eigen::MatrixXd lb, std::vector<int> nodes)
{
    return _bounds->setLowerBounds(lb, nodes);
}

bool Variable::setUpperBounds(Eigen::MatrixXd ub, std::vector<int> nodes)
{
    return _bounds->setUpperBounds(ub, nodes);
}

bool Variable::setBounds(Eigen::MatrixXd lb, Eigen::MatrixXd ub, std::vector<int> nodes)
{
    return _bounds->setBounds(lb, ub, nodes);
}

bool Variable::setInitialGuess(Eigen::MatrixXd initial_guess, std::vector<int> nodes)
{
    if (nodes.empty())
    {
        _initial_guess = initial_guess;
//        std::cout << "ig: \n" << initial_guess << std::endl;
    }
    else
    {
//        std::cout << "ig at nodes: " << std::endl;

//        for (int elem : nodes)
//        {
//            std::cout << elem << " ";
//        }
//        std::cout << std::endl;

//        std::cout << "with values: \n" << initial_guess << std::endl;

         for (int i = 0; i < nodes.size(); i++) {
             _initial_guess(nodes[i]) = initial_guess(i);
         }

    }


     return true;
}

std::vector<int> Variable::getNodes()
{
    return _nodes;
}

std::string Variable::getName()
{
    return _name;
}

int Variable::getDim()
{
    return _dim;
}

Eigen::MatrixXd Variable::getLowerBounds()
{
    return _bounds->getLowerBounds();
}

Eigen::MatrixXd Variable::getUpperBounds()
{
    return _bounds->getUpperBounds();
}

Eigen::MatrixXd Variable::getInitialGuess()
{
    return _initial_guess;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> Variable::getBounds()
{
    return _bounds->getBounds();
}

casadi::SX Variable::getSym()
{
    return _sym;
}

Parameter::Parameter(std::string name, int dim, int n_nodes):
    _name(name),
    _dim(dim),
    _n_nodes(n_nodes)
{
    _values.resize(_dim, _n_nodes);

    _sym = casadi::SX::sym(_name, _dim);
}

bool Parameter::setNodes(std::vector<int> nodes, bool erasing)
{
    _nodes = nodes;
    return true;
}

bool Parameter::assign(const Eigen::MatrixXd& values, const std::vector<int>& nodes)
{
    // Case 1: Replace entire matrix
    if (nodes.empty())
    {
        // If _values is already initialized, enforce size consistency
        if (_values.size() != 0)
        {
            if (_values.rows() != values.rows() ||
                _values.cols() != values.cols())
            {
                std::cerr << "assign error: dimension mismatch when replacing full matrix.\n"
                          << "Expected (" << _values.rows() << ", " << _values.cols()
                          << ") but got (" << values.rows() << ", " << values.cols() << ").\n";
                return false;
            }
        }

        _values = values;
        return true;
    }

    // Case 2: Set specific node columns

    // Guard 1: column count must match number of nodes
    if (values.cols() != static_cast<int>(nodes.size()))
    {
        std::cerr << "assign error: values.cols() must equal nodes.size().\n";
        return false;
    }

    // Guard 2: row count must match parameter dimension
    if (values.rows() != _values.rows())
    {
        std::cerr << "assign error: row mismatch.\n";
        return false;
    }

    // Guard 3: node indices must be valid
    for (int node : nodes)
    {
        if (node < 0 || node >= _values.cols())
        {
            std::cerr << "assign error: node index out of bounds: "
                      << node << "\n";
            return false;
        }
    }

    // Assignment
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i)
    {
        _values.col(nodes[i]) = values.col(i);
    }

    return true;
}

std::vector<int> Parameter::getNodes()
{
    return _nodes;
}

std::string Parameter::getName()
{
    return _name;
}

int Parameter::getDim()
{
    return _dim;
}

Eigen::MatrixXd Parameter::getValues()
{
    return _values;
}


casadi::SX Parameter::getSym()
{
    return _sym;
}

