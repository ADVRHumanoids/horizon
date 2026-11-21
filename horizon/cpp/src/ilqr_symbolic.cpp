#include "ilqr/casadi_function.h"
#include "ilqr/state_space.h"
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
bool matchingComponent(const StateSpace &space, const StateSpace &input_space,
                       const std::vector<int> &indices, int begin,
                       int storage_offset, int tangent_offset,
                       int &matched_tangent_offset)
{
    if (&space == &input_space &&
        begin + space.stateSize() <= int(indices.size()))
    {
        bool matches = true;
        for (int i = 0; i < space.stateSize(); ++i)
        {
            matches = matches && indices[begin + i] == storage_offset + i;
        }
        if (matches)
        {
            matched_tangent_offset = tangent_offset;
            return true;
        }
    }
    if (const auto *product = dynamic_cast<const ProductSpace *>(&space))
    {
        for (const auto &component : product->components())
        {
            if (matchingComponent(*component, input_space, indices, begin,
                                  storage_offset, tangent_offset,
                                  matched_tangent_offset))
            {
                return true;
            }
            storage_offset += component->stateSize();
            tangent_offset += component->tangentSize();
        }
    }
    return false;
}

int euclideanDirection(const StateSpace &space, int index, int storage_offset,
                       int tangent_offset)
{
    if (index < storage_offset || index >= storage_offset + space.stateSize())
    {
        return -2;
    }
    if (dynamic_cast<const EuclideanSpace *>(&space))
    {
        return tangent_offset + index - storage_offset;
    }
    if (const auto *product = dynamic_cast<const ProductSpace *>(&space))
    {
        for (const auto &component : product->components())
        {
            if (index >= storage_offset &&
                index < storage_offset + component->stateSize())
            {
                return euclideanDirection(*component, index, storage_offset,
                                          tangent_offset);
            }
            storage_offset += component->stateSize();
            tangent_offset += component->tangentSize();
        }
    }
    return -2;
}

bool nativeDirections(const StateSpace &input_space, const StateSpace &state,
                      int control_size, int graph_inputs,
                      const std::vector<int> &indices, int &cursor,
                      std::vector<int> &directions)
{
    if (const auto *product = dynamic_cast<const ProductSpace *>(&input_space))
    {
        for (const auto &component : product->components())
        {
            if (!nativeDirections(*component, state, control_size, graph_inputs,
                                  indices, cursor, directions))
            {
                return false;
            }
        }
        return true;
    }
    const int size = input_space.stateSize();
    if (cursor + size > int(indices.size()))
    {
        return false;
    }
    if (dynamic_cast<const EuclideanSpace *>(&input_space))
    {
        if (size == 0)
        {
            return true;
        }
        const int first = indices[cursor], ns = state.stateSize();
        for (int i = 0; i < size; ++i)
        {
            const int index = indices[cursor + i];
            if (index != first + i || index < 0 || index >= graph_inputs)
            {
                return false;
            }
            int direction = -1;
            if (index < ns)
            {
                direction = euclideanDirection(state, index, 0, 0);
                if (direction == -2)
                {
                    return false;
                }
            }
            else if (index < ns + control_size)
            {
                direction = state.tangentSize() + index - ns;
            }
            directions.push_back(direction);
        }
    }
    else
    {
        int offset = 0;
        if (!matchingComponent(state, input_space, indices, cursor, 0, 0,
                               offset))
        {
            return false;
        }
        for (int i = 0; i < input_space.tangentSize(); ++i)
        {
            directions.push_back(offset + i);
        }
    }
    cursor += size;
    return true;
}

class GraphFunction final : public Function,
                            public StageTangentLinearization,
                            public StageTangentQuadratic,
                            public SymbolicStageSource
{
  public:
    GraphFunction(int nin, std::vector<Operation> operations,
                  casadi::Function symbolic_stage)
        : Function(nin, operations.back().function->outputSize()),
          _operations(std::move(operations)),
          _symbolicStage(std::move(symbolic_stage))
    {
        int size = nin;
        _exact = true;
        for (const auto &op : _operations)
        {
            Workspace w;
            w.function = op.function->clone();
            w.offset = size;
            w.input.resize(op.function->inputSize());
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
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<GraphFunction>(inputSize(), _operations,
                                               _symbolicStage);
    }
    const casadi::Function &symbolicStage() const override
    {
        return _symbolicStage;
    }
    bool hasHessian() const override
    {
        return _exact;
    }
    void valueAndStageLinearization(Input input, const StateSpace &state_space,
                                    int control_size, MatrixInput retraction,
                                    VectorOutput value,
                                    MatrixOutput jac) override
    {
        const int ns = state_space.stateSize(), nt = state_space.tangentSize();
        if (control_size < 0 || input.size() != inputSize() ||
            input.size() < ns + control_size || value.size() != outputSize() ||
            jac.rows() != outputSize() || jac.cols() != nt + control_size ||
            retraction.rows() != ns || retraction.cols() != nt ||
            !input.allFinite() || !retraction.allFinite())
        {
            throw std::invalid_argument("graph stage linearization mismatch");
        }
        _stageDirections.setZero(inputSize(), nt + control_size);
        _stageDirections.topLeftCorner(ns, nt) = retraction;
        _stageDirections.block(ns, nt, control_size, control_size)
            .setIdentity();
        forwardDirectional(input, _stageDirections, value, jac, &state_space,
                           control_size);
        if (!value.allFinite() || !jac.allFinite())
        {
            throw EvaluationError("graph stage linearization is not finite");
        }
    }
    void valueAndStageQuadratic(Input input, const StateSpace &state,
                                int control_size, MatrixInput retraction,
                                VectorOutput value, VectorOutput gradient,
                                MatrixOutput hessian) override
    {
        const int ns = state.stateSize(), nt = state.tangentSize();
        const int nd = nt + control_size;
        if (!_exact || outputSize() != 1 || input.size() != inputSize() ||
            control_size < 0 || input.size() < ns + control_size ||
            retraction.rows() != ns || retraction.cols() != nt ||
            value.size() != 1 || gradient.size() != nd ||
            hessian.rows() != nd || hessian.cols() != nd ||
            !input.allFinite() || !retraction.allFinite())
        {
            throw std::invalid_argument("graph stage quadratic mismatch");
        }
        _values.head(inputSize()) = input;
        _directional.setZero(_values.size(), nd);
        _directional.topLeftCorner(ns, nt) = retraction;
        _directional.block(ns, nt, control_size, control_size).setIdentity();
        for (size_t i = 0; i < _work.size(); ++i)
        {
            auto &w = _work[i];
            const auto &op = _operations[i];
            for (size_t j = 0; j < op.inputs.size(); ++j)
            {
                w.input(j) = _values(op.inputs[j]);
            }
            auto *local =
                dynamic_cast<TangentLinearization *>(w.function.get());
            const auto *local_space =
                local ? local->localInputSpace() : nullptr;
            w.nativeDirections.clear();
            int cursor = 0;
            w.localQuadratic = op.native && local_space &&
                local->hasTangentHessian() &&
                !op.function->hasJacobianProduct() &&
                nativeDirections(*local_space, state, control_size, inputSize(),
                                 op.inputs, cursor, w.nativeDirections) &&
                cursor == int(op.inputs.size()) &&
                w.nativeDirections.size() == size_t(local_space->tangentSize());
            if (w.localQuadratic)
            {
                w.jac.resize(w.function->outputSize(),
                             local_space->tangentSize());
                local->valueAndTangentLinearization(
                    w.input,
                    _values.segment(w.offset, w.function->outputSize()), w.jac);
                auto result =
                    _directional.middleRows(w.offset, w.function->outputSize());
                result.setZero();
                for (int j = 0; j < local_space->tangentSize(); ++j)
                {
                    if (w.nativeDirections[j] >= 0)
                    {
                        result.col(w.nativeDirections[j]) += w.jac.col(j);
                    }
                }
            }
            else
            {
                w.jac.resize(w.function->outputSize(), w.input.size());
                w.inputJac.resize(w.input.size(), nd);
                for (size_t j = 0; j < op.inputs.size(); ++j)
                {
                    w.inputJac.row(j) = _directional.row(op.inputs[j]);
                }
                w.function->valueAndJacobian(
                    w.input,
                    _values.segment(w.offset, w.function->outputSize()), w.jac);
                _directional.middleRows(w.offset, w.function->outputSize())
                    .noalias() = w.jac * w.inputJac;
            }
        }
        value = _values.tail(1);
        _adjoint.setZero(_values.size());
        _adjoint.tail(1).setOnes();
        gradient.setZero();
        hessian.setZero();
        for (int i = int(_work.size()) - 1; i >= 0; --i)
        {
            auto &w = _work[i];
            const auto &indices = _operations[i].inputs;
            const int m = w.function->outputSize();
            const auto weights = _adjoint.segment(w.offset, m);
            if (w.localQuadratic)
            {
                auto *local =
                    dynamic_cast<TangentLinearization *>(w.function.get());
                const int n = int(w.nativeDirections.size());
                w.hess.resize(n, n);
                local->weightedTangentHessian(w.input, weights, w.hess);
                for (int a = 0; a < n; ++a)
                {
                    const int ia = w.nativeDirections[a];
                    if (ia < 0)
                    {
                        continue;
                    }
                    gradient(ia) += w.jac.col(a).dot(weights);
                    for (int b = 0; b < n; ++b)
                    {
                        const int ib = w.nativeDirections[b];
                        if (ib >= 0)
                        {
                            hessian(ia, ib) += w.hess(a, b);
                        }
                    }
                }
            }
            else
            {
                w.hess.resize(w.input.size(), w.input.size());
                w.function->weightedHessian(w.input, weights, w.hess);
                w.product.noalias() = w.hess * w.inputJac;
                hessian.noalias() += w.inputJac.transpose() * w.product;
                for (size_t j = 0; j < indices.size(); ++j)
                {
                    _adjoint(indices[j]) += w.jac.col(j).dot(weights);
                }
            }
        }
        gradient.head(nt).noalias() +=
            retraction.transpose() * _adjoint.head(ns);
        gradient.tail(control_size) += _adjoint.segment(ns, control_size);
        if (!_adjoint.head(ns).isZero(0))
        {
            _stageCurvature.resize(nt, nt);
            state.retractionHessian(input.head(ns), _adjoint.head(ns),
                                    _stageCurvature);
            hessian.topLeftCorner(nt, nt) += _stageCurvature;
        }
        if (!value.allFinite() || !gradient.allFinite() || !hessian.allFinite())
        {
            throw EvaluationError("graph stage quadratic is not finite");
        }
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
        std::vector<int> nativeDirections;
        bool localQuadratic = false;
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
        forwardDirectional(x, directions, value, product, nullptr, 0);
    }
    void forwardDirectional(Input x, MatrixInput directions, VectorOutput value,
                            MatrixOutput product, const StateSpace *state,
                            int control_size)
    {
        _values.head(inputSize()) = x;
        _directional.resize(_values.size(), directions.cols());
        _directional.topRows(inputSize()) = directions;
        for (size_t i = 0; i < _work.size(); ++i)
        {
            auto &w = _work[i];
            const auto &op = _operations[i];
            for (size_t j = 0; j < op.inputs.size(); ++j)
            {
                w.input(j) = _values(op.inputs[j]);
            }
            auto *local =
                dynamic_cast<TangentLinearization *>(w.function.get());
            const auto *local_space =
                local ? local->localInputSpace() : nullptr;
            _nativeDirections.clear();
            int cursor = 0;
            if (state && op.native && local_space &&
                !op.function->hasJacobianProduct() &&
                nativeDirections(*local_space, *state, control_size,
                                 inputSize(), op.inputs, cursor,
                                 _nativeDirections) &&
                cursor == int(op.inputs.size()) &&
                _nativeDirections.size() == size_t(local_space->tangentSize()))
            {
                _localJac.resize(w.function->outputSize(),
                                 local_space->tangentSize());
                local->valueAndTangentLinearization(
                    w.input,
                    _values.segment(w.offset, w.function->outputSize()),
                    _localJac);
                auto result =
                    _directional.middleRows(w.offset, w.function->outputSize());
                result.setZero();
                for (int j = 0; j < local_space->tangentSize(); ++j)
                {
                    if (_nativeDirections[j] >= 0)
                    {
                        result.col(_nativeDirections[j]) += _localJac.col(j);
                    }
                }
            }
            else
            {
                w.inputJac.resize(w.input.size(), directions.cols());
                for (size_t j = 0; j < op.inputs.size(); ++j)
                {
                    w.inputJac.row(j) = _directional.row(op.inputs[j]);
                }
                w.function->valueAndJacobianProduct(
                    w.input, w.inputJac,
                    _values.segment(w.offset, w.function->outputSize()),
                    _directional.middleRows(w.offset,
                                            w.function->outputSize()));
            }
        }
        value = _values.tail(outputSize());
        product = _directional.bottomRows(outputSize());
    }
    void weightedHessianImpl(Input x, Input weights, MatrixOutput hess) override
    {
        _derivative.resize(_values.size(), inputSize());
        _adjoint.resize(_values.size());
        _H.resize(_values.size(), _values.size());
        forward(x, true);
        _adjoint.setZero();
        _adjoint.tail(outputSize()) = weights;
        _H.setZero();
        for (int i = int(_work.size()) - 1; i >= 0; --i)
        {
            auto &w = _work[i];
            const auto &indices = _operations[i].inputs;
            const int off = w.offset, m = w.function->outputSize();
            w.hess.resize(w.input.size(), w.input.size());
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
    casadi::Function _symbolicStage;
    std::vector<Workspace> _work;
    Vector _values, _adjoint;
    Matrix _derivative, _H, _directional, _identity, _stageDirections,
        _localJac, _stageCurvature;
    std::vector<int> _nativeDirections;
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
    casadi::Function symbolic_stage;
    if (operations.size() == 1)
    {
        std::vector<std::string> input_names;
        for (const auto &input : inputs)
        {
            input_names.push_back(input.name());
        }
        symbolic_stage = casadi::Function(name + "_stage", inputs, {expression},
                                          input_names, {"f"});
    }
    auto function = std::make_shared<GraphFunction>(nin, std::move(operations),
                                                    std::move(symbolic_stage));
    if (exact && !function->hasHessian())
    {
        throw std::invalid_argument(
            "exact symbolic costs require Hessians for all native calls");
    }
    add(name, std::move(function), std::move(parameters));
}
} // namespace horizon::ilqr
