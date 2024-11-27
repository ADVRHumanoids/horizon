#ifndef HORIZON_PARSER_H
#define HORIZON_PARSER_H


#include <yaml-cpp/yaml.h>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <casadi/casadi.hpp>
#include <Eigen/Dense>

#include "wrapped_function.h"
#include "typedefs.h"

namespace horizon
{

class Problem
{
using Real=horizon::Real;
using MatrixXr=horizon::MatrixXr;
using VectorXr=horizon::VectorXr;

public:

    void from_yaml(YAML::Node problem_yaml);

    void print();

    void update_bounds();

    struct Variable
    {
        typedef std::shared_ptr<Variable> Ptr;

        std::string name;
        casadi::SX sym;
        MatrixXr lb, ub, value, initial_guess;

        int size();
    };

    struct Function
    {
        typedef std::shared_ptr<Function> Ptr;

        std::string name;
        casadi::Function fun;
        MatrixXr lb, ub;
        std::vector<int> nodes;
    };

    std::map<std::string, Variable::Ptr> var_map, param_map;

    std::map<std::string, Function::Ptr> fun_map, cost_map, constr_map;

    std::vector<Variable::Ptr> state_vec, input_vec;

    casadi::SX x, u;
    MatrixXr xlb, xub, ulb, uub, x_ini, u_ini;

    casadi::Function dynamics;

    casadi_utils::WrappedFunction inv_dyn;

    int N;
    Real dt;

private:

    Variable::Ptr yaml_to_variable(YAML::Node item);
    Function::Ptr yaml_to_function(std::pair<YAML::Node, YAML::Node> item,
                                   std::string outname);
};

}



#endif // HORIZON_PARSER_H
