#include "ilqr/casadi_function.h"
#include "ilqr/state_space.h"
#include <cmath>
#include <stdexcept>

namespace horizon::ilqr
{
namespace
{
bool symbolicRetraction(const StateSpace &space, const casadi::MX &x,
                        const casadi::MX &dx, casadi::MX &out)
{
    using casadi::MX;
    using casadi::Slice;
    if (dynamic_cast<const EuclideanSpace *>(&space))
    {
        out = x + dx;
        return true;
    }
    if (dynamic_cast<const QuaternionSpace *>(&space))
    {
        const auto s = MX::dot(dx, dx);
        const auto small = s < 1e-16;
        const auto theta = sqrt(MX::if_else(small, 1, s));
        const auto a = MX::if_else(small, .5 - s / 48, sin(theta / 2) / theta);
        const auto b =
            MX::if_else(small, 1 - s / 8 + s * s / 384, cos(theta / 2));
        const auto v = x(Slice(0, 3));
        const auto cross = MX::vertcat({v(1) * dx(2) - v(2) * dx(1),
                                        v(2) * dx(0) - v(0) * dx(2),
                                        v(0) * dx(1) - v(1) * dx(0)});
        const auto q = MX::vertcat(
            {b * v + a * (x(3) * dx + cross), b * x(3) - a * MX::dot(v, dx)});
        out = q / sqrt(MX::dot(q, q));
        return true;
    }
    if (const auto *product = dynamic_cast<const ProductSpace *>(&space))
    {
        std::vector<MX> parts;
        int ns = 0, nt = 0;
        for (const auto &component : product->components())
        {
            MX part;
            if (!symbolicRetraction(
                    *component, x(Slice(ns, ns + component->stateSize())),
                    dx(Slice(nt, nt + component->tangentSize())), part))
            {
                return false;
            }
            parts.push_back(part);
            ns += component->stateSize();
            nt += component->tangentSize();
        }
        out = MX::vertcat(parts);
        return true;
    }
    return false;
}

bool symbolicDifference(const StateSpace &space, const casadi::MX &from,
                        const casadi::MX &to, casadi::MX &out)
{
    using casadi::MX;
    using casadi::Slice;
    if (dynamic_cast<const EuclideanSpace *>(&space))
    {
        out = to - from;
        return true;
    }
    if (dynamic_cast<const QuaternionSpace *>(&space))
    {
        const auto a = from(Slice(0, 3)), b = to(Slice(0, 3));
        const auto cross =
            MX::vertcat({a(1) * b(2) - a(2) * b(1), a(2) * b(0) - a(0) * b(2),
                         a(0) * b(1) - a(1) * b(0)});
        auto v = from(3) * b - to(3) * a - cross;
        auto w = from(3) * to(3) + MX::dot(a, b);
        const auto sign = MX::if_else(w < 0, -1, 1);
        v *= sign;
        w *= sign;
        const auto s2 = MX::dot(v, v);
        const auto small = s2 < 1e-16;
        const auto s = sqrt(MX::if_else(small, 1, s2));
        const auto scale = MX::if_else(small, 2, 2 * atan2(s, w) / s);
        out = scale * v;
        return true;
    }
    if (const auto *product = dynamic_cast<const ProductSpace *>(&space))
    {
        std::vector<MX> parts;
        int ns = 0;
        for (const auto &component : product->components())
        {
            MX part;
            const auto segment = Slice(ns, ns + component->stateSize());
            if (!symbolicDifference(*component, from(segment), to(segment),
                                    part))
            {
                return false;
            }
            parts.push_back(part);
            ns += component->stateSize();
        }
        out = MX::vertcat(parts);
        return true;
    }
    return false;
}

int inputSize(const casadi::Function &f)
{
    if (f.is_null() || f.n_in() == 0 || f.n_out() != 1 || f.size2_out(0) != 1)
    {
        throw std::invalid_argument(
            "expected vector inputs and one vector output");
    }
    int n = 0;
    for (int i = 0; i < f.n_in(); ++i)
    {
        if (f.size2_in(i) != 1 || !f.sparsity_in(i).is_dense())
        {
            throw std::invalid_argument(
                "function inputs must be dense column vectors");
        }
        n += f.size1_in(i);
    }
    return n;
}

bool canExpandLocal(const casadi::Function &f, int directions)
{
    // Keep expansion bounded and exclude opaque calls or unsupported MX nodes.
    if ((!f.is_a("SXFunction") && !f.is_a("MXFunction")) || directions > 32 ||
        f.size1_out(0) > 32 || f.n_instructions() > 200)
    {
        return false;
    }
    int inputs = 0;
    for (int i = 0; i < f.n_in(); ++i)
    {
        inputs += f.size1_in(i);
    }
    if (inputs > 64)
    {
        return false;
    }
    casadi_int scalar_budget = 0;
    for (casadi_int i = 0; i < f.n_instructions(); ++i)
    {
        const auto op = f.instruction_id(i);
        if (op > casadi::OP_OUTPUT)
        {
            switch (op)
            {
            case casadi::OP_MTIMES:
            case casadi::OP_TRANSPOSE:
            case casadi::OP_DOT:
            case casadi::OP_HORZCAT:
            case casadi::OP_VERTCAT:
            case casadi::OP_DIAGCAT:
            case casadi::OP_RESHAPE:
            case casadi::OP_GETNONZEROS:
            case casadi::OP_PROJECT:
                break;
            default:
                return false;
            }
        }
        if (f.is_a("MXFunction") && op != casadi::OP_INPUT &&
            op != casadi::OP_OUTPUT)
        {
            const casadi_int entries = f.instruction_MX(i).numel();
            if (entries > 8192 - scalar_budget)
            {
                return false;
            }
            scalar_budget += entries;
        }
    }
    return true;
}
} // namespace

casadi::Function compileLocalDerivatives(const casadi::Function &f,
                                         const StateSpace &space,
                                         LocalDerivativeKind kind)
{
    using casadi::MX;
    using casadi::Slice;
    if (f.is_null() || f.n_in() < 2 || f.n_out() != 1 ||
        f.size1_in(0) != space.stateSize() || f.size2_in(0) != 1 ||
        f.size2_in(1) != 1 || f.size2_out(0) != 1 ||
        (kind == LocalDerivativeKind::Dynamics &&
         f.size1_out(0) != space.stateSize()) ||
        (kind == LocalDerivativeKind::ExactScalar && f.size1_out(0) != 1))
    {
        return {};
    }
    std::vector<MX> args;
    for (int i = 0; i < f.n_in(); ++i)
    {
        if (f.size2_in(i) != 1 || !f.sparsity_in(i).is_dense())
        {
            return {};
        }
        args.push_back(
            MX::sym("local_input_" + std::to_string(i), f.size1_in(i)));
    }
    const int nt = space.tangentSize(), nu = f.size1_in(1);
    auto delta = MX::sym("local_delta", nt + nu);
    MX retracted;
    if (!symbolicRetraction(space, args[0], delta(Slice(0, nt)), retracted))
    {
        return {};
    }
    auto perturbed = args;
    perturbed[0] = retracted;
    perturbed[1] = args[1] + delta(Slice(nt, nt + nu));
    auto y = f(perturbed).at(0);
    auto value = f(args).at(0);
    MX differentiated = y;
    if (kind == LocalDerivativeKind::Dynamics &&
        !symbolicDifference(space, value, y, differentiated))
    {
        return {};
    }
    auto J = MX::jacobian(differentiated, delta);
    auto zero = MX::zeros(nt + nu);
    auto J0 = MX::substitute(J, delta, zero);
    std::vector<MX> outputs{value, J0};
    if (kind == LocalDerivativeKind::ExactScalar)
    {
        outputs.push_back(MX::substitute(MX::hessian(y, delta), delta, zero));
    }
    else if (kind == LocalDerivativeKind::GaussNewton)
    {
        outputs.push_back(MX::mtimes(J0.T(), J0));
    }
    casadi::Function result(
        f.name() + "_local", args, outputs, f.name_in(),
        kind == LocalDerivativeKind::Jacobian ||
                kind == LocalDerivativeKind::Dynamics
            ? std::vector<std::string>{"value", "jac"}
            : std::vector<std::string>{"value", "jac", "hess"});
    return canExpandLocal(f, nt + nu) ? result.expand() : result;
}

CasadiFunction::CasadiFunction(const casadi::Function &f, bool exact_hessian)
    : Function(ilqr::inputSize(f), f.size1_out(0)),
      _exact_hessian(exact_hessian)
{
    using casadi::MX;
    auto z = MX::sym("z", Function::inputSize());
    std::vector<MX> inputs;
    int offset = 0;
    for (int i = 0; i < f.n_in(); ++i)
    {
        inputs.push_back(z(casadi::Slice(offset, offset + f.size1_in(i))));
        offset += f.size1_in(i);
    }
    auto y = f(inputs).at(0);
    _value = casadi::Function(f.name() + "_value", {z}, {y});
    _jac = casadi::Function(f.name() + "_jacobian", {z}, {MX::jacobian(y, z)});
    _linear = casadi::Function(f.name() + "_linearization", {z},
                               {y, MX::jacobian(y, z)});
    std::vector<casadi_int> unused;
    _linear.function().sparsity_out(0).get_triplet(_valueRows, unused);
    _linear.function().sparsity_out(1).get_triplet(_jacRows, _jacCols);
    if (exact_hessian)
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
void CasadiFunction::valueAndJacobianImpl(Input x, VectorOutput v,
                                          MatrixOutput j)
{
    _linear.setInput(0, x);
    _linear.call();
    v = _linear.getOutput(0);
    j = _linear.getOutput(1);
}
void CasadiFunction::valueAndJacobianProductImpl(Input x,
                                                 MatrixInput directions,
                                                 VectorOutput value,
                                                 MatrixOutput product)
{
    _linear.setInput(0, x);
    _linear.callRaw();
    const auto &v = _linear.getOutputNonzeros(0);
    const auto &j = _linear.getOutputNonzeros(1);
    value.setZero();
    product.setZero();
    for (size_t k = 0; k < v.size(); ++k)
    {
        if (!std::isfinite(v[k]))
        {
            throw EvaluationError("CasADi value is not finite");
        }
        value(_valueRows[k]) = v[k];
    }
    for (double entry : j)
    {
        if (!std::isfinite(entry))
        {
            throw EvaluationError("CasADi Jacobian is not finite");
        }
    }
    for (int col = 0; col < directions.cols(); ++col)
    {
        for (size_t k = 0; k < j.size(); ++k)
        {
            product(_jacRows[k], col) += j[k] * directions(_jacCols[k], col);
        }
    }
}

} // namespace horizon::ilqr
