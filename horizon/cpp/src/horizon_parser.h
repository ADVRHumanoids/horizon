#ifndef HORIZON_PARSER_H
#define HORIZON_PARSER_H


#include <yaml-cpp/yaml.h>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "wrapped_function.h"

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


class Function
{
 public:

     typedef std::shared_ptr<Function> Ptr;

     Function(casadi::Function fun, int n_nodes);

     virtual bool setNodes(std::vector<int> nodes, bool erasing) = 0;

     std::vector<int> getNodes();
     std::string getName();
     int getDim();
     casadi::Function getFunction();

 protected:

     int _n_nodes;
     std::string _name;
     std::vector<int> _nodes;
     int _dim;
     casadi::Function _fun;

 };

class Constraint: public Function
{
 public:

    typedef std::shared_ptr<Constraint> Ptr;

    Constraint(casadi::Function fun, int n_nodes);

    bool setNodes(std::vector<int> nodes, bool erasing) override;

    bool setLowerBounds(Eigen::MatrixXd lb, std::vector<int> nodes = {});
    bool setUpperBounds(Eigen::MatrixXd ub, std::vector<int> nodes = {});

    Eigen::MatrixXd getLowerBounds();
    Eigen::MatrixXd getUpperBounds();
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> getBounds();

private:

    Bounds::Ptr _bounds;


};

class Cost: public Function
{
 public:

    typedef std::shared_ptr<Cost> Ptr;

    Cost(casadi::Function fun, int n_nodes);

    bool setNodes(std::vector<int> nodes, bool erasing) override;

};


class Problem
{

public:

    void from_yaml(YAML::Node problem_yaml);

    void print();

    void update_bounds();

    std::map<std::string, Variable::Ptr> var_map;

    std::map<std::string, Parameter::Ptr> param_map;

    std::map<std::string, Function::Ptr> fun_map;
    std::map<std::string, Cost::Ptr> cost_map;
    std::map<std::string, Constraint::Ptr> constr_map;

    std::vector<Variable::Ptr> state_vec, input_vec;

    casadi::SX x, u;
    Eigen::MatrixXd xlb, xub, ulb, uub, x_ini, u_ini;

    casadi::Function dynamics;

    casadi_utils::WrappedFunction inv_dyn;

    int N;
    double dt;

private:

    Variable::Ptr yaml_to_variable(std::pair<YAML::Node, YAML::Node> item);
    Parameter::Ptr yaml_to_parameter(std::pair<YAML::Node, YAML::Node> item);
    casadi::Function make_casadi_function(std::pair<YAML::Node, YAML::Node> item, std::string outname);
    Constraint::Ptr yaml_to_constraint(std::pair<YAML::Node, YAML::Node> item);
    Cost::Ptr yaml_to_cost(std::pair<YAML::Node, YAML::Node> item);



};

}



#endif // HORIZON_PARSER_H
