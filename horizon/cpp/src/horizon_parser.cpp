#include "horizon_parser.h"

using namespace horizon;

Problem::Variable::Ptr Problem::yaml_to_variable(YAML::Node item)
{
    auto v = std::make_shared<Variable>();

    auto var_data = item;
    auto name = var_data["name"].as<std::string>();
    v->name = name;

    int size = var_data["size"].as<int>();
    v->sym = casadi::SX::sym(name, size);

    try
    {
        auto lb = var_data["lb"].as<std::vector<Real>>();
        auto ub = var_data["ub"].as<std::vector<Real>>();
        auto ini = var_data["initial_guess"].as<std::vector<Real>>();

        v->lb = MatrixXr::Map(lb.data(), size, lb.size()/size);
        v->ub = MatrixXr::Map(ub.data(), size, ub.size()/size);
        v->initial_guess = MatrixXr::Map(ini.data(), size, ini.size()/size);
    }
    catch(YAML::Exception&)
    {

    }

    try
    {
        auto values = var_data["values"].as<std::vector<Real>>();
        v->value = MatrixXr::Map(values.data(), size, values.size()/size);
    }
    catch(YAML::Exception&)
    {

    }

    return v;
}

Problem::Function::Ptr Problem::yaml_to_function(std::pair<YAML::Node, YAML::Node> item,
                                                 std::string outname)
{
    auto fun = std::make_shared<Function>();
    auto name = item.first.as<std::string>();
    auto var_data = item.second;

    fun->name = name;
    fun->nodes = var_data["nodes"].as<std::vector<int>>();

    // wrap f as f(x, y, p...)
    auto f = casadi::Function::deserialize(var_data["function"].as<std::string>());

    // compute function value as SX
    std::vector<casadi::SX> fun_input;
    std::vector<casadi::SX> fun_input_param;
    std::vector<std::string> param_names;
    for(int i = 0; i < f.n_in(); i++)
    {
        fun_input.push_back(var_map.at(f.name_in(i))->sym);

        if(param_map.count(f.name_in(i)))
        {
            param_names.push_back(f.name_in(i));
            fun_input_param.push_back(fun_input.back());
        }
    }

    // compute symbolic value
    casadi::SX fn_value = f(fun_input)[0];

    // wrap function with required signature
    std::vector<casadi::SX> wrapped_inputs = {x, u};
    wrapped_inputs.insert(wrapped_inputs.end(),
                          fun_input_param.begin(),
                          fun_input_param.end());

    std::vector<std::string> wrapped_names = {"x", "u"};
    wrapped_names.insert(wrapped_names.end(),
                         param_names.begin(),
                         param_names.end());

    fun->fun = casadi::Function(f.name(),
                                wrapped_inputs, {fn_value},
                                wrapped_names, {outname});

    // bounds
    try
    {
        auto lb = var_data["lb"].as<std::vector<Real>>();
        auto ub = var_data["ub"].as<std::vector<Real>>();

        fun->lb = MatrixXr::Map(lb.data(), f.size1_out(0), lb.size()/f.size1_out(0));
        fun->ub = MatrixXr::Map(ub.data(), f.size1_out(0), ub.size()/f.size1_out(0));
    }
    catch(YAML::Exception& e)
    {

    }

    return fun;

}

void Problem::from_yaml(YAML::Node problem_yaml)
{
    // nodes
    N = problem_yaml["n_nodes"].as<int>();

    // dt
    try
    {
        dt = problem_yaml["solver"]["dt"].as<Real>();
        std::cout << "problem_yaml[solver][dt].as<Real>() = " << dt << "\n";
    }
    catch(YAML::Exception& e)
    {
        dt = -1.0;
    }

    // inv dyn
    inv_dyn = casadi::Function::deserialize(problem_yaml["inv_dyn"].as<std::string>());

    // retrieve state
    std::vector<casadi::SX> x_sx_vec;
    for(auto item : problem_yaml["state"])
    {
        auto v = yaml_to_variable(item);

        std::cout << "state " << v->name << "[" << v->sym.size1() << "]\n" <<
                     "=======\n";

        state_vec.push_back(v);
        var_map[v->name] = v;
        x_sx_vec.push_back(v->sym);

    }

    // retrieve input
    std::vector<casadi::SX> u_sx_vec;
    for(auto item : problem_yaml["input"])
    {
        auto v = yaml_to_variable(item);

        std::cout << "input " << v->name << "[" << v->sym.size1() << "]\n" <<
                     "=======\n";

        input_vec.push_back(v);
        var_map[v->name] = v;
        u_sx_vec.push_back(v->sym);

    }

    // retrieve param
    for(auto item : problem_yaml["param"])
    {
        auto v = yaml_to_variable(item);

        std::cout << "param " << v->name << "[" << v->sym.size1() << "]\n" <<
                     "=======\n";

        var_map[v->name] = param_map[v->name] = v;
    }

    // dynamics
    dynamics = casadi::Function::deserialize(problem_yaml["dynamics"].as<std::string>());
    x = casadi::SX::vertcat(x_sx_vec);
    u = casadi::SX::vertcat(u_sx_vec);

    // retrive costs
    for(auto item : problem_yaml["cost"])
    {
        auto f = yaml_to_function(item, "l");

        cost_map[f->name] = fun_map[f->name] = f;
    }

    // retrive constraints
    for(auto item : problem_yaml["constraint"])
    {
        auto f = yaml_to_function(item, "h");

        constr_map[f->name] = fun_map[f->name] = f;
    }

    // compute state and input bounds
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
        xlb.middleRows(row, xi->size()) = xi->lb;
        xub.middleRows(row, xi->size()) = xi->ub;
        x_ini.middleRows(row, xi->size()) = xi->initial_guess;
        row += xi->size();
    }

    row = 0;
    for(auto ui : input_vec)
    {
        ulb.middleRows(row, ui->size()) = ui->lb;
        uub.middleRows(row, ui->size()) = ui->ub;
        u_ini.middleRows(row, ui->size()) = ui->initial_guess;
        row += ui->size();
    }
}

int Problem::Variable::size()
{
    return sym.size1();
}
