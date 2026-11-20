#include "ilqr/casadi_function.h"
#include <atomic>
#include <set>

namespace horizon::ilqr
{
namespace
{
struct Operation
{
    std::shared_ptr<const Function> function;
    std::vector<int> inputs;
    bool native = false;
};

// Pure selection/concatenation needs no numerical CasADi operation. Prove
// this symbolically at setup; never infer it from sampled derivatives.
bool selection(const casadi::MX &expression,
               const std::vector<casadi::MX> &symbols,
               const std::vector<int> &indices, std::vector<int> &selected)
{
    const auto input = casadi::MX::vertcat(symbols);
    const auto J = casadi::MX::jacobian(expression, input);
    if (casadi::MX::depends_on(J, input))
    {
        return false;
    }
    const auto offset = casadi::MX::substitute(
        expression, input, casadi::MX::zeros(input.size1()));
    if (!casadi::MX::symvar(offset).empty() || !casadi::MX::symvar(J).empty())
    {
        return false;
    }
    const auto zero = casadi::MX::evalf(offset);
    if (!zero.is_zero())
    {
        return false;
    }
    const auto jac = casadi::MX::evalf(J);
    const auto sp = jac.sparsity();
    std::vector<int> columns(expression.size1(), -1);
    const auto nonzeros = jac.nonzeros();
    for (int c = 0; c < sp.size2(); ++c)
    {
        for (auto k = sp.colind(c); k < sp.colind(c + 1); ++k)
        {
            if (nonzeros[k] == 0)
            {
                continue;
            }
            const int r = sp.row(k);
            if (nonzeros[k] != 1 || columns[r] != -1)
            {
                return false;
            }
            columns[r] = c;
        }
    }
    for (int col : columns)
    {
        if (col < 0)
        {
            return false;
        }
    }
    for (int col : columns)
    {
        selected.push_back(indices[col]);
    }
    return true;
}

Operation symbolicOperation(const std::string &name,
                            const casadi::MX &expression,
                            const std::vector<casadi::MX> &symbols,
                            const std::vector<int> &indices, bool exact)
{
    const auto needed = casadi::MX::symvar(expression);
    std::vector<casadi::MX> inputs;
    std::vector<int> selected;
    std::size_t offset = 0;
    for (const auto &symbol : symbols)
    {
        bool used = false;
        for (const auto &dependency : needed)
        {
            if (casadi::MX::is_equal(symbol, dependency))
            {
                used = true;
                break;
            }
        }
        if (used)
        {
            inputs.push_back(symbol);
            selected.insert(selected.end(), indices.begin() + offset,
                            indices.begin() + offset + symbol.size1());
        }
        offset += symbol.size1();
    }
    // Keep one declared input for constant maps: Function dimensions are
    // positive, and CasADi supplies its structurally zero derivative.
    if (inputs.empty())
    {
        inputs.push_back(symbols.front());
        selected.assign(indices.begin(),
                        indices.begin() + symbols.front().size1());
    }
    return {std::make_shared<CasadiFunction>(
                casadi::Function(name, inputs, {expression}), exact),
            std::move(selected)};
}

// A numerical tape: all native workspaces are cloned, and symbolic segments
// contain only pure CasADi. Shared calls are evaluated once per forward sweep.
class GraphFunction final : public Function
{
  public:
    GraphFunction(int nin, std::vector<Operation> operations)
        : Function(nin, operations.back().function->outputSize()),
          _operations(std::move(operations))
    {
        int size = nin;
        _exact = true;
        for (const auto &op : _operations)
        {
            Workspace w;
            w.function = op.function->clone();
            w.offset = size;
            w.input.resize(op.function->inputSize());
            if (op.function->hasHessian())
            {
                w.hess.resize(op.function->inputSize(),
                              op.function->inputSize());
            }
            w.inputJac.resize(op.function->inputSize(), nin);
            for (int index : op.inputs)
            {
                if (index < 0 || index >= size)
                {
                    throw std::invalid_argument("invalid graph dependency");
                }
            }
            if (op.inputs.size() != size_t(op.function->inputSize()))
            {
                throw std::invalid_argument("graph input dimension mismatch");
            }
            size += op.function->outputSize();
            _exact = _exact && op.function->hasHessian();
            _work.push_back(std::move(w));
        }
        _values.resize(size);
        // The ambient derivative/adjoint and quadratic tapes are used only
        // by exact Hessians. First-derivative graphs use the seeded tape.
        if (_exact)
        {
            _derivative.resize(size, nin);
            _adjoint.resize(size);
            _H.resize(size, size);
        }
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<GraphFunction>(inputSize(), _operations);
    }
    bool hasHessian() const override
    {
        return _exact;
    }
    void setEvaluationContext(
        const std::shared_ptr<EvaluationContext> &context) override
    {
        for (size_t i = 0; i < _work.size(); ++i)
        {
            if (_operations[i].native)
            {
                if (context)
                {
                    _work[i].function = context->share(_operations[i].function);
                }
                else
                {
                    _work[i].function = _operations[i].function->clone();
                }
            }
            _work[i].function->setEvaluationContext(context);
        }
    }

  private:
    struct Workspace
    {
        std::shared_ptr<Function> function;
        int offset;
        Vector input;
        Matrix jac, hess, inputJac, cross, product;
    };
    void forward(Input x, bool derivatives)
    {
        _values.head(inputSize()) = x;
        if (derivatives)
        {
            _derivative.topRows(inputSize()).setIdentity();
        }
        for (size_t i = 0; i < _work.size(); ++i)
        {
            auto &w = _work[i];
            const auto &op = _operations[i];
            for (size_t j = 0; j < op.inputs.size(); ++j)
            {
                w.input(j) = _values(op.inputs[j]);
            }
            if (derivatives)
            {
                w.jac.resize(w.function->outputSize(), w.function->inputSize());
                w.function->valueAndJacobian(
                    w.input,
                    _values.segment(w.offset, w.function->outputSize()), w.jac);
                w.inputJac.resize(w.input.size(), inputSize());
                for (size_t j = 0; j < op.inputs.size(); ++j)
                {
                    w.inputJac.row(j) = _derivative.row(op.inputs[j]);
                }
                _derivative.middleRows(w.offset, w.function->outputSize())
                    .noalias() = w.jac * w.inputJac;
            }
            else
            {
                w.function->evaluate(
                    w.input,
                    _values.segment(w.offset, w.function->outputSize()));
            }
        }
    }
    void evaluateImpl(Input x, VectorOutput value) override
    {
        forward(x, false);
        value = _values.tail(outputSize());
    }
    void jacobianImpl(Input x, MatrixOutput jac) override
    {
        if (_identity.rows() != inputSize())
        {
            _identity.setIdentity(inputSize(), inputSize());
        }
        valueAndJacobianProductImpl(x, _identity, _values.tail(outputSize()),
                                    jac);
    }
    void valueAndJacobianImpl(Input x, VectorOutput value,
                              MatrixOutput jac) override
    {
        if (_identity.rows() != inputSize())
        {
            _identity.setIdentity(inputSize(), inputSize());
        }
        valueAndJacobianProductImpl(x, _identity, value, jac);
    }
    void valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                     VectorOutput value,
                                     MatrixOutput product) override
    {
        _values.head(inputSize()) = x;
        _directional.resize(_values.size(), directions.cols());
        _directional.topRows(inputSize()) = directions;
        for (size_t i = 0; i < _work.size(); ++i)
        {
            auto &w = _work[i];
            const auto &op = _operations[i];
            w.inputJac.resize(w.input.size(), directions.cols());
            for (size_t j = 0; j < op.inputs.size(); ++j)
            {
                w.input(j) = _values(op.inputs[j]);
                w.inputJac.row(j) = _directional.row(op.inputs[j]);
            }
            w.function->valueAndJacobianProduct(
                w.input, w.inputJac,
                _values.segment(w.offset, w.function->outputSize()),
                _directional.middleRows(w.offset, w.function->outputSize()));
        }
        value = _values.tail(outputSize());
        product = _directional.bottomRows(outputSize());
    }
    void weightedHessianImpl(Input x, Input weights, MatrixOutput hess) override
    {
        forward(x, true);
        _adjoint.setZero();
        _adjoint.tail(outputSize()) = weights;
        _H.setZero();
        for (int i = int(_work.size()) - 1; i >= 0; --i)
        {
            auto &w = _work[i];
            const auto &indices = _operations[i].inputs;
            const int off = w.offset, m = w.function->outputSize();
            w.function->weightedHessian(w.input, _adjoint.segment(off, m),
                                        w.hess);
            w.cross.noalias() = w.jac.transpose() * _H.block(off, 0, m, off);
            w.product.noalias() = _H.block(off, off, m, m) * w.jac;
            w.hess.noalias() += w.jac.transpose() * w.product;
            for (size_t a = 0; a < indices.size(); ++a)
            {
                const int ia = indices[a];
                _adjoint(ia) += w.jac.col(a).dot(_adjoint.segment(off, m));
                for (int b = 0; b < off; ++b)
                {
                    _H(ia, b) += w.cross(a, b);
                    _H(b, ia) += w.cross(a, b);
                }
                for (size_t b = 0; b < indices.size(); ++b)
                {
                    _H(ia, indices[b]) += w.hess(a, b);
                }
            }
        }
        hess = _H.topLeftCorner(inputSize(), inputSize());
    }
    std::vector<Operation> _operations;
    std::vector<Workspace> _work;
    Vector _values, _adjoint;
    Matrix _derivative, _H, _directional, _identity;
    bool _exact;
};
} // namespace

SymbolicNativeFunction
SymbolicFunctionManager::addNative(const std::string &name,
                                   std::shared_ptr<const Function> function)
{
    add(name, function);
    SymbolicNativeFunction result;
    result._name = name;
    result._function = std::move(function);
    result._calls = _calls;
    return result;
}
casadi::MX SymbolicNativeFunction::operator()(const casadi::MX &argument) const
{
    if (argument.size2() != 1 || argument.size1() != _function->inputSize())
    {
        throw std::invalid_argument("native symbolic call input size mismatch");
    }
    static std::atomic<unsigned long> id{0};
    auto result =
        casadi::MX::sym("native_" + _name + "_" + std::to_string(id++),
                        _function->outputSize());
    _calls->push_back({argument, result, _function});
    return result;
}

void SymbolicFunctionManager::addExpression(
    const std::string &name, const std::vector<casadi::MX> &inputs,
    const casadi::MX &expression, bool exact)
{
    if (inputs.size() < 2 || expression.size2() != 1 || expression.size1() == 0)
    {
        throw std::invalid_argument("stage expressions require state/control "
                                    "inputs and a vector output");
    }
    int nin = 0;
    std::vector<Parameter> parameters;
    std::set<std::string> inputNames;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        const auto &x = inputs[i];
        if (!x.is_symbolic() || x.size2() != 1 || !x.sparsity().is_dense() ||
            !inputNames.insert(x.name()).second)
        {
            throw std::invalid_argument(
                "stage inputs must be distinct dense vector symbols");
        }
        nin += x.size1();
        if (i >= 2)
        {
            parameters.push_back({x.name(), int(x.size1())});
        }
    }
    // Only include native calls reachable from this expression. Native call
    // symbols have globally unique names, but reachability uses symbol
    // identity.
    std::vector<bool> used(_calls->size(), false);
    std::vector<casadi::MX> needed = casadi::MX::symvar(expression);
    for (int i = int(_calls->size()) - 1; i >= 0; --i)
    {
        const auto &call = _calls->at(i);
        for (const auto &symbol : needed)
        {
            if (casadi::MX::is_equal(symbol, call.result))
            {
                used[i] = true;
                break;
            }
        }
        if (used[i])
        {
            auto symbols = casadi::MX::symvar(call.argument);
            needed.insert(needed.end(), symbols.begin(), symbols.end());
        }
    }
    std::vector<casadi::MX> symbols = inputs;
    std::vector<int> indices;
    for (int i = 0; i < nin; ++i)
    {
        indices.push_back(i);
    }
    int offset = nin;
    std::vector<Operation> operations;
    for (size_t i = 0; i < _calls->size(); ++i)
    {
        if (!used[i])
        {
            continue;
        }
        const auto &call = _calls->at(i);
        std::vector<int> nativeIndices;
        if (!selection(call.argument, symbols, indices, nativeIndices))
        {
            operations.push_back(
                symbolicOperation(name + "_arg_" + std::to_string(i),
                                  call.argument, symbols, indices, exact));
            for (int j = 0; j < operations.back().function->outputSize(); ++j)
            {
                nativeIndices.push_back(offset++);
            }
        }
        operations.push_back({call.function, std::move(nativeIndices), true});
        for (int j = 0; j < call.function->outputSize(); ++j)
        {
            indices.push_back(offset++);
        }
        symbols.push_back(call.result);
    }
    operations.push_back(symbolicOperation(name + "_outer", expression, symbols,
                                           indices, exact));
    auto function = std::make_shared<GraphFunction>(nin, std::move(operations));
    if (exact && !function->hasHessian())
    {
        throw std::invalid_argument(
            "exact symbolic costs require Hessians for all native calls");
    }
    add(name, std::move(function), std::move(parameters));
}
} // namespace horizon::ilqr
