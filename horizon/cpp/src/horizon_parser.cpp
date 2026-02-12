#include "horizon_parser.h"

using namespace horizon;

Variable::Ptr Problem::yaml_to_variable(std::pair<YAML::Node, YAML::Node> item)
{


    auto name = item.first.as<std::string>();
    auto var_data = item.second;

    int size = var_data["size"].as<int>();

    auto var = std::make_shared<Variable>(name, size, N);


    try
    {
        auto lb_yaml = var_data["lb"].as<std::vector<double>>();
        auto ub_yaml = var_data["ub"].as<std::vector<double>>();
        auto ini_yaml = var_data["initial_guess"].as<std::vector<double>>();

        std::cout << "setting bounds of variable " << name << ": " << std::endl;

        auto lb = Eigen::MatrixXd::Map(lb_yaml.data(), size, lb_yaml.size()/size);
        var->setLowerBounds(lb);

        auto ub = Eigen::MatrixXd::Map(ub_yaml.data(), size, ub_yaml.size()/size);
        var->setUpperBounds(ub);

        auto ini = Eigen::MatrixXd::Map(ini_yaml.data(), size, ini_yaml.size()/size);
        var->setInitialGuess(ini);

        std::cout << "lb: " << std::endl << lb << std::endl;
        std::cout << "ub: " << std::endl << ub << std::endl;
        std::cout << "ini: " << std::endl << ini << std::endl;

    }
    catch(YAML::Exception&)
    {

    }

    return var;
}

Parameter::Ptr Problem::yaml_to_parameter(std::pair<YAML::Node, YAML::Node> item)
{

    auto name = item.first.as<std::string>();
    auto var_data = item.second;

    int size = var_data["size"].as<int>();

    auto var = std::make_shared<Parameter>(name, size, N+1);
    var->setNodes(var_data["nodes"].as<std::vector<int>>(), true);

    try
    {
        auto values_yaml = var_data["values"].as<std::vector<double>>();
        auto values = Eigen::MatrixXd::Map(values_yaml.data(), size, values_yaml.size()/size);

        var->setValues(values, var->getNodes());
    }
    catch(YAML::Exception&)
    {

    }

    return var;
}

casadi::Function Problem::make_casadi_function(std::pair<YAML::Node, YAML::Node> item, std::string outname)
{
    auto name = item.first.as<std::string>();
    auto var_data = item.second;

//    std::cout << "name: " << name << std::endl;

    // wrap f as f(x, y, p...)
    auto f = casadi::Function::deserialize(var_data["function"].as<std::string>());

    // compute function value as SX
    std::vector<casadi::SX> fun_input;
    std::vector<casadi::SX> fun_input_param;
    std::vector<std::string> param_names;

    for(auto var_name_yaml : var_data["var_depends"])
    {
        auto var_name = var_name_yaml.as<std::string>();
//        std::cout << "adding variable: " << var_name << std::endl;

        fun_input.push_back(var_map.at(var_name)->getSym());
//        std::cout << "success: " << var_map.at(var_name)->getSym() << std::endl;
    }

    for(auto par_name_yaml : var_data["param_depends"])
    {
        auto par_name = par_name_yaml.as<std::string>();
//        std::cout << "adding parameter: " << par_name << std::endl;

        param_names.push_back(par_name);
        fun_input_param.push_back(param_map.at(par_name)->getSym());
//        std::cout << "success: " << param_map.at(par_name)->getSym() << std::endl;
    }

    // compute symbolic value

    std::vector<casadi::SX> total_input = fun_input;

    total_input.insert(total_input.end(), fun_input_param.begin(), fun_input_param.end());

    casadi::SX fn_value = f(total_input)[0];

//    std::cout << "function: " << fn_value << std::endl;

    // wrap function with required signature
    std::vector<casadi::SX> wrapped_inputs = {x, u};
    wrapped_inputs.insert(wrapped_inputs.end(),
                          fun_input_param.begin(),
                          fun_input_param.end());

    std::vector<std::string> wrapped_names = {"x", "u"};
    wrapped_names.insert(wrapped_names.end(),
                         param_names.begin(),
                         param_names.end());


    auto casadi_function = casadi::Function(f.name(),
                                wrapped_inputs, {fn_value},
                                wrapped_names, {outname});

    std::cout << "function: " << name << std::endl;
    std::cout << "    inputs: " << wrapped_inputs << std::endl;
    std::cout << "       fun: " << fn_value << std::endl;

//    std::cout << "function created" << std::endl;

    return casadi_function;


}

Constraint::Ptr Problem::yaml_to_constraint(std::pair<YAML::Node, YAML::Node> item)
{

    auto name = item.first.as<std::string>();
    auto var_data = item.second;

    auto casadi_function = make_casadi_function(item, "h");
    auto fun = std::make_shared<Constraint>(casadi_function, N);


    fun->setNodes(var_data["nodes"].as<std::vector<int>>(), true);

    // bounds
    try
    {
        auto lb_yaml = var_data["lb"].as<std::vector<double>>();
        auto ub_yaml = var_data["ub"].as<std::vector<double>>();

        auto lb = Eigen::MatrixXd::Map(lb_yaml.data(), fun->getFunction().size1_out(0), lb_yaml.size()/fun->getFunction().size1_out(0));
        auto ub = Eigen::MatrixXd::Map(ub_yaml.data(), fun->getFunction().size1_out(0), ub_yaml.size()/fun->getFunction().size1_out(0));

//        std::cout << "setting bounds of constraint " << name << ": " << std::endl;
        fun->setLowerBounds(lb);
        fun->setUpperBounds(ub);

    }
    catch(YAML::Exception& e)
    {

    }

    return fun;

}

Cost::Ptr Problem::yaml_to_cost(std::pair<YAML::Node, YAML::Node> item)
{
    auto name = item.first.as<std::string>();
    auto var_data = item.second;


    auto casadi_function = make_casadi_function(item, "l");

    auto fun = std::make_shared<Cost>(casadi_function, N);

    fun->setNodes(var_data["nodes"].as<std::vector<int>>(), true);



    return fun;
}

Residual::Ptr Problem::yaml_to_residual(std::pair<YAML::Node, YAML::Node> item)
{
    auto name = item.first.as<std::string>();
    auto var_data = item.second;

    auto casadi_function = make_casadi_function(item, "res");

    auto fun = std::make_shared<Residual>(casadi_function, N);
    fun->setNodes(var_data["nodes"].as<std::vector<int>>(), true);

    return fun;
}


void Problem::from_yaml(YAML::Node problem_yaml)
{
    // nodes
    N = problem_yaml["n_nodes"].as<int>();
    std::vector<double> dt_vec = problem_yaml["dt"].as<std::vector<double>>();
    dt = Eigen::Map<Eigen::VectorXd>(dt_vec.data(), dt_vec.size());

    std::cout << "N: " << N << "\n";
    std::cout << "dt: " << dt.transpose() << "\n";

    std::cout << "------------------\n";


//    // inv dyn
//    inv_dyn = casadi::Function::deserialize(problem_yaml["inv_dyn"].as<std::string>());

    // retrieve state
    for (auto item : problem_yaml["state"])
    {
        auto v = yaml_to_variable(item);
        var_map[v->getName()] = v;
    }

    std::vector<casadi::SX> x_sx_vec;

    for (const auto& name_node : problem_yaml["state_list"])
    {
        std::string name = name_node.as<std::string>();

        auto it = var_map.find(name);
        if (it == var_map.end())
        {
            throw std::runtime_error(
                "State '" + name + "' listed in state_list but not defined in state");
        }

        state_vec.push_back(it->second);           // ordered
        x_sx_vec.push_back(it->second->getSym());  // ordered
    }

    // retrieve input
    for (auto item : problem_yaml["input"])
    {
        auto v = yaml_to_variable(item);
        var_map[v->getName()] = v;
    }

    std::vector<casadi::SX> u_sx_vec;

    for (const auto& name_node : problem_yaml["input_list"])
    {
        std::string name = name_node.as<std::string>();

        auto it = var_map.find(name);
        if (it == var_map.end())
        {
            throw std::runtime_error(
                "Input '" + name + "' listed in input_list but not defined in state");
        }

        input_vec.push_back(it->second);           // ordered
        u_sx_vec.push_back(it->second->getSym());  // ordered
    }

    // retrieve param
    for(auto item : problem_yaml["param"])
    {

//        std::cout << "adding param: " << std::endl;

        auto v = yaml_to_parameter(item);

//        std::cout << v->getName() << " -- size: [" << v->getSym().size1() << "]" << std::endl;

        param_map[v->getName()] = v;

//        std::cout << "-------------------------------------------" << std::endl;
    }

    // dynamics

    if (problem_yaml["dynamics"])
    {
        dynamics = casadi::Function::deserialize(problem_yaml["dynamics"].as<std::string>());
    }
    else
    {
        std::cerr << "'dynamics' key not found in problem_yaml. " << std::endl;
    }

    x = casadi::SX::vertcat(x_sx_vec);
    u = casadi::SX::vertcat(u_sx_vec);
    update_bounds();

    // retrive costs
    for(auto item : problem_yaml["cost"])
    {

         std::cout << "----- creating cost function: -------" << std::endl;

        auto f = yaml_to_cost(item);

        cost_map[f->getName()] = f;
    }
    // retrive residuals
    for(auto item : problem_yaml["residual"])
    {

//         std::cout << "----- creating residual function: -------" << std::endl;

        auto f = yaml_to_residual(item);

        residual_map[f->getName()] = f;

//        std::cout << "-------------------------------------------" << std::endl;
    }

    // retrive constraints
    for(auto item : problem_yaml["constraint"])
    {

//        std::cout << "----- creating constraint function: -------" << std::endl;

        auto f = yaml_to_constraint(item);

        constr_map[f->getName()]= f;

//        std::cout << "-------------------------------------------" << std::endl;
    }

//     compute state and input bounds
    update_bounds();
}

void Problem::print()
{
    std::cout << "n_nodes = " << N << "\n\n";
    std::cout << "dynamics: " << dynamics << "\n\n";
    std::cout << "state = " << x << "\n\n";
    std::cout << "input = " << u << "\n\n";
}

void Problem::update_bounds()
{
    xlb.resize(x.size1(), N+1);
    xub.resize(x.size1(), N+1);
    ulb.resize(u.size1(), N);
    uub.resize(u.size1(), N);
    x_ini.resize(x.size1(), N+1);
    u_ini.resize(u.size1(), N);

    int row = 0;

    for(auto xi : state_vec)
    {
        xlb.middleRows(row, xi->getDim()) = xi->getLowerBounds();
        xub.middleRows(row, xi->getDim()) = xi->getUpperBounds();
        x_ini.middleRows(row, xi->getDim()) = xi->getInitialGuess();
        row += xi->getDim();
    }

    row = 0;
    for(auto ui : input_vec)
    {
        ulb.middleRows(row, ui->getDim()) = ui->getLowerBounds();
        uub.middleRows(row, ui->getDim()) = ui->getUpperBounds();
        u_ini.middleRows(row, ui->getDim()) = ui->getInitialGuess();
        row += ui->getDim();
    }
}

//int Problem::Variable::size()
//{
//    return sym.size1();
//}
