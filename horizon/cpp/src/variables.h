#ifndef VARIABLES_H
#define VARIABLES_H

#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "bounds.h"

namespace horizon
{

class Variable
{
public:

    typedef std::shared_ptr<Variable> Ptr;

    Variable(std::string name, int dim, int n_nodes);
    bool setNodes(std::vector<int> nodes, bool erasing);
    bool setLowerBounds(Eigen::MatrixXd lb, std::vector<int> nodes = {});
    bool setUpperBounds(Eigen::MatrixXd ub, std::vector<int> nodes = {});
    bool setInitialGuess(Eigen::MatrixXd initial_guess, std::vector<int> nodes = {});

    Eigen::MatrixXd getValues();

    std::vector<int> getNodes();
    std::string getName();
    int getDim();
    Eigen::MatrixXd getLowerBounds();
    Eigen::MatrixXd getUpperBounds();

    Eigen::MatrixXd getInitialGuess();



    casadi::SX getSym();


private:

    std::string _name;
    int _dim;
    int _n_nodes;
    Bounds::Ptr _bounds;
    std::vector<int> _nodes;
    Eigen::MatrixXd _initial_guess;

    casadi::SX _sym;

};

class Parameter
{
public:

    typedef std::shared_ptr<Parameter> Ptr;

    Parameter(std::string name, int dim, int n_nodes);
    bool setNodes(std::vector<int> nodes, bool erasing);
    bool setValues(const Eigen::MatrixXd& values, const std::vector<int>& nodes = {});

    Eigen::MatrixXd getValues();

    std::vector<int> getNodes();
    std::string getName();
    int getDim();

    casadi::SX getSym();


private:

    std::string _name;
    int _dim;
    int _n_nodes;
    std::vector<int> _nodes;
    Eigen::MatrixXd _values;

    casadi::SX _sym;

};
}
#endif // VARIABLES_H
