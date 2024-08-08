#include "sqp.h"
#include <casadi/casadi.hpp>
#include "typedefs.h"

int main()
{
//    auto x = casadi::SX::sym("x", 2);
//    std::cout<<"x: "<<x<<std::endl;
//    auto u = casadi::SX::sym("u", 1);
//    std::cout<<"u: "<<u<<std::endl;

//    auto f = 3.*x(0)*x(1)+ 2.*u;
//    auto g = casadi::SX::vertcat({x, u});

//    casadi::DM x0(3,1);
//    x0(0) = 2.;
//    x0(1) = 3.;
//    x0(2) = 4.;

//    casadi::DM lb(3,1), ub(3,1);
//    lb(0) = 1.5;
//    lb(1) = -1.;
//    lb(1) = -2.;
//    ub(0) = 1.5;
//    ub(1) = 2.;
//    ub(1) = 2.;


//    casadi::Dict opts;
//    opts["max_iter"] = 5;
//    //opts["printLevel"] = "none";

//    /// CONSTRUCTOR 1
//    std::cout<<"USE MAIN CONSTRUCTOR"<<std::endl;
//    horizon::SQPGaussNewton<casadi::SX> sqp("sqp", "osqp", f, g, casadi::SX::vertcat({x, u}), opts);

//    auto solution = sqp.solve(x0, lb, ub, lb, ub);

//    std::cout<<"solution: "<<solution["x"]<<"   f: "<<solution["f"]<<"  g: "<<solution["g"]<<std::endl;

//    casadi::DMVector var_trj = sqp.getVariableTrajectory();
//    std::vector<Real> objs = sqp.getObjectiveIterations();
//    std::vector<Real> cons = sqp.getConstraintNormIterations();
//    for(unsigned int i = 0; i < sqp.getNumberOfIterations(); ++i)
//        std::cout<<"iter "<<i<<"-> sol: "<<var_trj[i]<<"    obj: "<<objs[i]<<"  cons: "<<cons[i]<<std::endl;

//    /// CONSTRUCTOR 2
//    std::cout<<"USE SECOND CONSTRUCTOR"<<std::endl;
//    auto F = casadi::Function("f", {casadi::SX::vertcat({x,u})}, {f}, {"x"}, {"f"});
//    auto G = casadi::Function("g", {casadi::SX::vertcat({x,u})}, {g}, {"x"}, {"g"});

//    horizon::SQPGaussNewton<casadi::SX> sqp2("sqp2", "qpoases", F, G, opts);

//    auto solution2 = sqp2.solve(x0, lb, ub, lb, ub);

//    std::cout<<"solution2: "<<solution2["x"]<<"   f: "<<solution2["f"]<<"  g: "<<solution2["g"]<<std::endl;

//    casadi::DMVector var_trj2 = sqp2.getVariableTrajectory();
//    std::vector<Real> objs2 = sqp2.getObjectiveIterations();
//    std::vector<Real> cons2 = sqp2.getConstraintNormIterations();
//    for(unsigned int i = 0; i < sqp2.getNumberOfIterations(); ++i)
//        std::cout<<"iter "<<i<<"-> sol2: "<<var_trj2[i]<<"    obj2: "<<objs2[i]<<"  cons: "<<cons2[i]<<std::endl;


}
