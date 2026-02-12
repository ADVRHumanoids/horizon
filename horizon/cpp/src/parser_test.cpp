#include "horizon_parser.h"
#include "ilqr.h"

int main()
{

//    YAML::Node config = YAML::LoadFile("/home/fruscelli/forest_ws/src/horizon/horizon/tests/data.yaml");
//    YAML::Node config = YAML::LoadFile("/home/fruscelli/forest_ws/src/horizon/horizon/tests/test_problem_save_0.yaml");
//    YAML::Node config = YAML::LoadFile("/home/fruscelli/forest_ws/src/horizon/horizon/tests/test_problem_save_1.yaml");
//    YAML::Node config = YAML::LoadFile("/home/fruscelli/forest_ws/src/horizon/horizon/tests/test_problem_save_2.yaml");
//    YAML::Node config = YAML::LoadFile("/home/fruscelli/forest_ws/src/horizon/horizon/tests/test_problem_save_3.yaml");
    YAML::Node config = YAML::LoadFile("/home/fruscelli/.ros/receding_kyon.yaml");

    // Set formatting globally, equivalent to np.set_printoptions(suppress=True, precision=4)
    std::cout << std::fixed << std::setprecision(4);


    horizon::Problem problem;
    problem.from_yaml(config);



//    horizon::IterativeLQR::OptionDict options = {};
    horizon::IterativeLQR::OptionDict options = {
        {"ilqr.constraint_violation_threshold", 1e-2},          // double
        {"ilqr.suppress_all_output", std::string("yes")},       // string
        {"ilqr.codegen_enabled", 1},                         // bool
        {"ilqr.codegen_workdir", std::string("/tmp/tyhio")},    // string
        {"ilqr.enable_gn", true},                               // bool
        {"ilqr.hxx_reg_base", 0.0},                             // double
        {"ilqr.n_threads", 0},                                  // int
    };

//    self.prb.getIntegrator(), self.N, self.opts
    horizon::IterativeLQR ilqr(problem.dynamics, problem.N, options);

    ilqr.setStateInitialGuess(problem.x_ini);
    ilqr.setInputInitialGuess(problem.u_ini);

    std::cout << "xini: " << problem.x_ini << std::endl;
    std::cout << "uini: " << problem.u_ini << std::endl;

    // set constraints
    for (auto constr_pair : problem.constr_map)
    {
        std::string name = constr_pair.first;
        auto constr = constr_pair.second; //std::dynamic_pointer_cast<horizon::Constraint>(constr_pair.second);

        std::cout << "adding to ilqr constraint: " << name << " ("<< constr->getFunction() << ")" << std::endl;
        std::cout << "at nodes: " << constr->getNodes() << std::endl;

        ilqr.setConstraint(constr->getNodes(), constr->getFunction());
    }

    // set costs
    for (auto cost_pair : problem.cost_map)
    {
        std::string name = cost_pair.first;
        auto cost = cost_pair.second; //std::dynamic_pointer_cast<horizon::Cost>();

        std::cout << "adding to ilqr cost: " << name << " ("<< cost->getFunction() << ")" << std::endl;
        std::cout << "at nodes: " << cost->getNodes() << std::endl;

        ilqr.setCost(cost->getNodes(), cost->getFunction());
    }

    // set residuals
    for (auto residual_pair : problem.residual_map)
    {
        std::string name = residual_pair.first;
        auto residual = residual_pair.second;

        std::cout << "adding to ilqr residual: " << name << " ("<< residual->getFunction() << ")" << std::endl;
        std::cout << "at nodes: " << residual->getNodes() << std::endl;

        ilqr.setResidual(residual->getNodes(), residual->getFunction());
    }


    // set bounds
    std::cout << "setting bounds: " << std::endl;

    std::cout << "xlb: \n" << problem.xlb << std::endl;
    std::cout << "xub: \n" << problem.xub << std::endl;
    std::cout << "ulb: \n" << problem.ulb << std::endl;
    std::cout << "uub: \n" << problem.uub << std::endl;


    ilqr.setStateBounds(problem.xlb, problem.xub);
    ilqr.setInputBounds(problem.ulb, problem.uub);

    // before each solve

    std::cout << "========== setting parameters values ===================" << std::endl;

    // set param
    for (auto param_pair : problem.param_map)
    {
        std::string name = param_pair.first;
        std::cout << "parameter: '" << name << "'" << std::endl;

        auto param = param_pair.second;

        Eigen::MatrixXd p_vals_temp = param->getValues();
        std::vector<int> nodes = param->getNodes();
        int dim = param->getDim();

        // allocate dim x (N+1)
        Eigen::MatrixXd p_vals(dim, problem.N + 1);

        // fill with NaN
        p_vals.setConstant(std::numeric_limits<double>::quiet_NaN());


        for (int j = 0; j < (int)nodes.size(); ++j) {
            int col = nodes[j];
            p_vals.col(col) = p_vals_temp.col(j);
        }

        std::cout << "nodes:  \n" << nodes << std::endl;
        std::cout << "values: \n" << p_vals << std::endl;
        std::cout << " -------------- " << std::endl;

        // send to ilqr
        ilqr.setParameterValue(name, p_vals);

    }

    std::cout << "=============================" << std::endl;

    // set dt
    std::cout << "setting dt: " << problem.dt.transpose() << std::endl;
    ilqr.setParameterValue("dt", problem.dt.transpose());


    // set nodes cost
    for (auto cost_pair : problem.cost_map)
    {
        ilqr.setIndices(cost_pair.first, cost_pair.second->getNodes());
        std::cout << "updating nodes of cost: '" << cost_pair.first << "': \n " << cost_pair.second->getNodes() << std::endl;
    }

    // set nodes residual
    for (auto residual_pair : problem.residual_map)
    {
        ilqr.setIndices(residual_pair.first, residual_pair.second->getNodes());
        std::cout << "updating nodes of residual: '" << residual_pair.first << "': \n " << residual_pair.second->getNodes() << std::endl;
    }

    // set nodes constraint
    for (auto constraint_pair : problem.constr_map)
    {
        ilqr.setIndices(constraint_pair.first, constraint_pair.second->getNodes());
        std::cout << "updating nodes of constraint: '" << constraint_pair.first << "': \n " << constraint_pair.second->getNodes() << std::endl;
    }


//    ilqr.setFinalConstraint();
//    ilqr.setFinalCost();

    ilqr.updateIndices();

    std::cout << "solve: " << std::endl;
    ilqr.setIterationCallback([&](const horizon::IterativeLQR::ForwardPassResult& res){ res.print(); return true; });
    ilqr.solve(100);

    std::unordered_map<std::string, Eigen::MatrixXd> solution_dict;

    // Get optimal trajectories
    Eigen::MatrixXd x_opt = ilqr.getStateTrajectory();
    Eigen::MatrixXd u_opt = ilqr.getInputTrajectory();

    int off = 0;
    for (const auto& var : problem.state_vec)
    {
        std::string vname = var->getName();
        int dim = var->getDim();   // or var->getSize()

        // slice x_opt[off:off+dim, :]
        solution_dict[vname] = x_opt.block(off, 0, dim, x_opt.cols());

        off += dim;
    }

    // ---- inputs ----
    off = 0;
    for (const auto& var : problem.input_vec)
    {
        std::string vname = var->getName();
        int dim = var->getDim();   // or var->getSize()

        // slice u_opt[off:off+dim, :]
        solution_dict[vname] = u_opt.block(off, 0, dim, u_opt.cols());

        off += dim;
    }

    // store full trajectories too
    solution_dict["x_opt"] = x_opt;
    solution_dict["u_opt"] = u_opt;

//    std::cout << ilqr.getConstraintsValues() << std::endl;
//    std::cout << ilqr.getCostsValues() << std::endl;


    for (const auto& kv : solution_dict)
    {
      const std::string& name = kv.first;
      const Eigen::MatrixXd& mat = kv.second;

      std::cout << "===== " << name << " =====\n";
      std::cout << "shape: " << mat.rows() << " x " << mat.cols() << "\n";

      int maxCols = std::min<int>(mat.cols(), 50);
      std::cout << mat.leftCols(maxCols) << "\n\n";
    }

    // Reset formatting if needed
    std::cout << std::defaultfloat;






}


