#include "ilqr/casadi_function.h"
#include <stdexcept>
#include <cmath>

namespace horizon::ilqr
{
namespace
{
int inputSize(const casadi::Function& f)
{
    if(f.is_null() || f.n_in() == 0 || f.n_out() != 1 || f.size2_out(0) != 1)
        throw std::invalid_argument("expected vector inputs and one vector output");
    int n = 0;
    for(int i = 0; i < f.n_in(); ++i)
    {
        if(f.size2_in(i) != 1 || !f.sparsity_in(i).is_dense())
            throw std::invalid_argument("function inputs must be dense column vectors");
        n += f.size1_in(i);
    }
    return n;
}
}

CasadiFunction::CasadiFunction(const casadi::Function& f, bool exact_hessian):
    Function(ilqr::inputSize(f), f.size1_out(0)), _exact_hessian(exact_hessian)
{
    using casadi::MX;
    auto z = MX::sym("z", Function::inputSize());
    std::vector<MX> inputs;
    int offset = 0;
    for(int i = 0; i < f.n_in(); ++i)
    {
        inputs.push_back(z(casadi::Slice(offset, offset + f.size1_in(i))));
        offset += f.size1_in(i);
    }
    auto y = f(inputs).at(0);
    _value = casadi::Function(f.name() + "_value", {z}, {y});
    _jac = casadi::Function(f.name() + "_jacobian", {z}, {MX::jacobian(y, z)});
    _linear = casadi::Function(f.name() + "_linearization", {z}, {y,MX::jacobian(y,z)});
    std::vector<casadi_int> unused;
    _linear.function().sparsity_out(0).get_triplet(_valueRows,unused);
    _linear.function().sparsity_out(1).get_triplet(_jacRows,_jacCols);
    if(exact_hessian)
    {
        auto w = MX::sym("weights", outputSize());
        _hess = casadi::Function(f.name() + "_hessian", {z, w},
                                {MX::hessian(MX::dot(w, y), z)});
    }
}
std::unique_ptr<Function> CasadiFunction::clone() const
{
    // WrappedFunction's copy constructor creates fresh numerical workspaces.
    return std::unique_ptr<Function>(new CasadiFunction(*this));
}
void CasadiFunction::evaluateImpl(Input x, VectorOutput v)
{
    _value.setInput(0, x);
    _value.call();
    v = _value.getOutput(0);
}
void CasadiFunction::jacobianImpl(Input x, MatrixOutput j)
{
    _jac.setInput(0, x);
    _jac.call();
    j = _jac.getOutput(0);
}
void CasadiFunction::weightedHessianImpl(Input x, Input w, MatrixOutput h)
{
    _hess.setInput(0, x);
    _hess.setInput(1, w);
    _hess.call();
    h = _hess.getOutput(0);
}
void CasadiFunction::valueAndJacobianImpl(Input x, VectorOutput v, MatrixOutput j)
{
    _linear.setInput(0,x); _linear.call();
    v=_linear.getOutput(0); j=_linear.getOutput(1);
}
void CasadiFunction::valueAndJacobianProductImpl(Input x,MatrixInput directions,VectorOutput value,MatrixOutput product)
{
    _linear.setInput(0,x);_linear.callRaw();
    const auto& v=_linear.getOutputNonzeros(0);
    const auto& j=_linear.getOutputNonzeros(1);
    value.setZero();product.setZero();
    for(size_t k=0;k<v.size();++k)
    {
        if(!std::isfinite(v[k])) throw EvaluationError("CasADi value is not finite");
        value(_valueRows[k])=v[k];
    }
    for(double entry:j) if(!std::isfinite(entry)) throw EvaluationError("CasADi Jacobian is not finite");
    for(int col=0;col<directions.cols();++col)
        for(size_t k=0;k<j.size();++k)
            product(_jacRows[k],col)+=j[k]*directions(_jacCols[k],col);
}

}
