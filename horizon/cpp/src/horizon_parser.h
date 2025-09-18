#ifndef HORIZON_PARSER_H
#define HORIZON_PARSER_H


#include <yaml-cpp/yaml.h>
#include <vector>
#include <boost/lexical_cast.hpp>

#include "wrapped_function.h"
#include "variables.h"
#include "functions.h"

namespace horizon
{

class Problem
{

public:

    void from_yaml(YAML::Node problem_yaml);

    void print();

    void update_bounds();

    std::map<std::string, horizon::Variable::Ptr> var_map;

    std::map<std::string, horizon::Parameter::Ptr> param_map;

    std::map<std::string, horizon::Function::Ptr> fun_map;
    std::map<std::string, horizon::Cost::Ptr> cost_map;
    std::map<std::string, horizon::Constraint::Ptr> constr_map;

    std::vector<horizon::Variable::Ptr> state_vec, input_vec;

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
