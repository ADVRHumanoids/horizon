#include "horizon_parser.h"
#include "ilqr.h"

int main()
{

    YAML::Node config = YAML::LoadFile("/home/fruscelli/forest_ws/src/horizon/horizon/tests/data.yaml");

    horizon::Problem problem;
    problem.from_yaml(config);


//    self.prb.getIntegrator(), self.N, self.opts
    horizon::IterativeLQR ilqr(problem.dynamics, problem.N);

    ilqr.setStateInitialGuess(problem.x_ini);
    ilqr.setInputInitialGuess(problem.u_ini);

    // set constraints
    for (auto constr_pair : problem.constr_map)
    {
        std::string name = constr_pair.first;
        auto constr = constr_pair.second; //std::dynamic_pointer_cast<horizon::Constraint>(constr_pair.second);

        std::cout << "adding to ilqr constraint: " << name << std::endl;
        std::cout << "at nodes: " << constr->getNodes() << std::endl;

        ilqr.setConstraint(constr->getNodes(), constr->getFunction());
    }

    // set costs
    for (auto cost_pair : problem.cost_map)
    {
        std::string name = cost_pair.first;
        auto cost = cost_pair.second; //std::dynamic_pointer_cast<horizon::Cost>();

        std::cout << "adding to ilqr cost: " << name << std::endl;
        std::cout << "at nodes: " << cost->getNodes() << std::endl;

        ilqr.setCost(cost->getNodes(), cost->getFunction());
    }

    // set bounds

    std::cout << "xlb: \n" << problem.xlb << std::endl;
    std::cout << "xub: \n" << problem.xub << std::endl;
    std::cout << "ulb: \n" << problem.ulb << std::endl;
    std::cout << "uub: \n" << problem.uub << std::endl;


    ilqr.setStateBounds(problem.xlb, problem.xub);
    ilqr.setInputBounds(problem.ulb, problem.uub);

    // before each solve

    // set param
    for (auto param_pair : problem.param_map)
    {
        std::string name = param_pair.first;
        auto param = param_pair.second;
        ilqr.setParameterValue(name, param->getValues());

        std::cout << "setting parameter '" << name <<  "' to value: \n" << param->getValues() << std::endl;
    }

    // set dt

    // set nodes cost
    for (auto cost_pair : problem.cost_map)
    {
        ilqr.setIndices(cost_pair.first, cost_pair.second->getNodes());
        std::cout << "updating nodes of cost: '" << cost_pair.first << "': \n " << cost_pair.second->getNodes() << std::endl;
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

    ilqr.solve(100);









}


