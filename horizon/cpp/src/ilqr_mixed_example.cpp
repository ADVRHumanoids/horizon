#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"
#include <cmath>
#include <iostream>

using namespace horizon::ilqr;

// A one-link Cartesian FK. A Pinocchio implementation uses the same interface,
// with a fresh Data object in clone(); nq != nv maps derive from TangentFunction.
class PlanarFK final : public Function
{
public:
    PlanarFK():Function(1,2) {}
    std::unique_ptr<Function> clone() const override { return std::make_unique<PlanarFK>(); }
    bool hasHessian() const override { return true; }
private:
    void evaluateImpl(Input q,VectorOutput p) override { p<<std::cos(q(0)),std::sin(q(0)); }
    void jacobianImpl(Input q,MatrixOutput j) override { j<<-std::sin(q(0)),std::cos(q(0)); }
    void weightedHessianImpl(Input q,Input w,MatrixOutput h) override
    { h(0,0)=-w(0)*std::cos(q(0))-w(1)*std::sin(q(0)); }
};

int main()
{
    using casadi::MX;
    SymbolicFunctionManager functions;
    auto fk=functions.addNative("tool_position",std::make_shared<PlanarFK>());
    auto q=MX::sym("q"),u=MX::sym("u"),ref=MX::sym("p_ref",2),e=MX::sym("error",2);
    casadi::Function cartesianError("cartesianError",{e},{10*e});

    functions.addExpression("dynamics",{q,u},q+u,false);
    functions.addResidual("effort",{q,u},0.1*u);
    functions.addResidual("tracking",{q,u,ref},cartesianError({fk(q)-ref}).at(0));

    const int N=10;
    horizon::IterativeLQR solver(functions,"dynamics",N,
        {{"ilqr.n_threads",2},{"ilqr.merit_der_threshold",1e-10}});
    std::vector<int> nodes;
    for(int k=0;k<N;++k) nodes.push_back(k);
    solver.setResidual(nodes,functions,"effort");
    solver.setResidual({N},functions,"tracking");
    Matrix reference(2,N+1);
    reference.row(0).setConstant(std::cos(1.0)); reference.row(1).setConstant(std::sin(1.0));
    solver.setParameterValue("p_ref",reference);
    Matrix lb=Matrix::Constant(1,N+1,-std::numeric_limits<double>::infinity()),ub=-lb;
    lb(0,0)=ub(0,0)=0;
    solver.setStateBounds(lb,ub);
    solver.solve(30);
    std::cout<<"Final joint angle: "<<solver.state(N)(0)<<" rad\n";
    return std::abs(solver.state(N)(0)-1.0)<1e-5 ? 0:1;
}
