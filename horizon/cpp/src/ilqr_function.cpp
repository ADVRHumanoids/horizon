#include "ilqr/function.h"
#include "ilqr/state_space.h"
#include <cstring>
#include <set>
#include <stdexcept>

namespace horizon::ilqr
{
namespace
{
void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::invalid_argument(message);
    }
}
} // namespace

Function::Function(int nin, int nout) : _input_size(nin), _output_size(nout)
{
    require(nin > 0 && nout > 0, "function dimensions must be positive");
}

void Function::evaluate(Input x, VectorOutput value)
{
    require(x.size() == inputSize() && value.size() == outputSize(),
            "function evaluation size mismatch");
    require(x.allFinite(), "function input is not finite");
    evaluateImpl(x, value);
    if (!value.allFinite())
    {
        throw EvaluationError("function value is not finite");
    }
}

void Function::jacobian(Input x, MatrixOutput jac)
{
    require(x.size() == inputSize() && jac.rows() == outputSize() &&
                jac.cols() == inputSize(),
            "function Jacobian size mismatch");
    require(x.allFinite(), "function input is not finite");
    jacobianImpl(x, jac);
    if (!jac.allFinite())
    {
        throw EvaluationError("function Jacobian is not finite");
    }
}

void Function::valueAndJacobianProduct(Input x, MatrixInput directions,
                                       VectorOutput value, MatrixOutput product)
{
    require(x.size() == inputSize() && directions.rows() == inputSize() &&
                value.size() == outputSize() &&
                product.rows() == outputSize() &&
                product.cols() == directions.cols(),
            "function directional Jacobian size mismatch");
    require(x.allFinite() && directions.allFinite(),
            "function directional input is not finite");
    valueAndJacobianProductImpl(x, directions, value, product);
    if (!value.allFinite() || !product.allFinite())
    {
        throw EvaluationError(
            "function directional linearization is not finite");
    }
}
void Function::valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                           VectorOutput value,
                                           MatrixOutput product)
{
    _productJacobian.resize(outputSize(), inputSize());
    valueAndJacobianImpl(x, value, _productJacobian);
    if (!_productJacobian.allFinite())
    {
        throw EvaluationError("function Jacobian is not finite");
    }
    multiplyJacobian(_productJacobian, directions, product);
}

void Function::multiplyJacobian(MatrixInput jac, MatrixInput directions,
                                MatrixOutput product)
{
    // Seeds for selection/concatenation and manifold retractions are mostly
    // zero. Preserve every nonzero, including subnormal chain-rule entries.
    if ((directions.array() != 0).count() <= directions.size() / 4)
    {
        product.setZero();
        for (int c = 0; c < directions.cols(); ++c)
        {
            for (int r = 0; r < directions.rows(); ++r)
            {
                if (directions(r, c) != 0)
                {
                    product.col(c).noalias() += directions(r, c) * jac.col(r);
                }
            }
        }
    }
    else
    {
        product.noalias() = jac * directions;
    }
}

namespace
{
class CachedFunction final : public Function, public TangentLinearization
{
  public:
    explicit CachedFunction(const Function &source)
        : Function(source.inputSize(), source.outputSize()),
          _source(source.clone()), _input(inputSize()), _value(outputSize())
    {
        _tangent = dynamic_cast<TangentFunction *>(_source.get());
        _jac.resize(outputSize(),
                    _tangent ? _tangent->tangentInputSpace().tangentSize()
                             : inputSize());
    }
    std::unique_ptr<Function> clone() const override
    {
        return std::make_unique<CachedFunction>(*_source);
    }
    bool hasHessian() const override
    {
        return _source->hasHessian();
    }
    const StateSpace *localInputSpace() const override
    {
        return _tangent ? &_tangent->tangentInputSpace() : nullptr;
    }
    bool hasTangentHessian() const override
    {
        return _tangent && _tangent->hasTangentHessian();
    }
    void weightedTangentHessian(Input x, Input weights,
                                MatrixOutput hess) override
    {
        if (!_tangent)
        {
            throw std::logic_error("cached tangent Hessian unavailable");
        }
        prepare(x);
        _tangent->weightedTangentHessian(x, weights, hess);
    }
    void valueAndTangentLinearization(Input x, VectorOutput value,
                                      MatrixOutput jac) override
    {
        if (!_tangent || x.size() != inputSize() ||
            value.size() != outputSize() || jac.rows() != outputSize() ||
            jac.cols() != _jac.cols() || !x.allFinite())
        {
            throw std::invalid_argument(
                "cached tangent linearization mismatch");
        }
        linearize(x);
        value = _value;
        jac = _jac;
    }
    void setEvaluationContext(
        const std::shared_ptr<EvaluationContext> &context) override
    {
        _source->setEvaluationContext(context);
    }
    void invalidate()
    {
        _hasInput = _hasValue = _hasJac = false;
    }

  protected:
    void evaluateImpl(Input x, VectorOutput value) override
    {
        prepare(x);
        if (!_hasValue)
        {
            _source->evaluate(x, _value);
            _hasValue = true;
        }
        value = _value;
    }
    void jacobianImpl(Input x, MatrixOutput jac) override
    {
        linearize(x);
        if (_tangent)
        {
            _projection.resize(_jac.cols(), inputSize());
            _tangent->tangentInputSpace().tangentProjection(x, _projection);
            jac.noalias() = _jac * _projection;
        }
        else
        {
            jac = _jac;
        }
    }
    void valueAndJacobianImpl(Input x, VectorOutput value,
                              MatrixOutput jac) override
    {
        jacobianImpl(x, jac);
        value = _value;
    }
    void weightedHessianImpl(Input x, Input weights, MatrixOutput hess) override
    {
        prepare(x);
        _source->weightedHessian(x, weights, hess);
    }
    void valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                     VectorOutput value,
                                     MatrixOutput product) override
    {
        if (_source->hasJacobianProduct())
        {
            prepare(x);
            _hasValue = false;
            _source->valueAndJacobianProduct(x, directions, _value, product);
            _hasValue = true;
            value = _value;
            return;
        }
        linearize(x);
        value = _value;
        if (_tangent)
        {
            _directions.resize(_jac.cols(), directions.cols());
            _tangent->tangentInputSpace().projectDirections(x, directions,
                                                            _directions);
            multiplyJacobian(_jac, _directions, product);
        }
        else
        {
            multiplyJacobian(_jac, directions, product);
        }
    }

  private:
    void prepare(Input x)
    {
        if (!_hasInput ||
            std::memcmp(_input.data(), x.data(), x.size() * sizeof(double)))
        {
            _input = x;
            _hasInput = true;
            _hasValue = _hasJac = false;
        }
    }
    void linearize(Input x)
    {
        prepare(x);
        if (_hasJac)
        {
            return;
        }
        _hasValue = false;
        if (_tangent)
        {
            _tangent->valueAndTangentLinearization(x, _value, _jac);
        }
        else
        {
            _source->valueAndJacobian(x, _value, _jac);
        }
        _hasValue = _hasJac = true;
    }
    std::unique_ptr<Function> _source;
    Vector _input, _value;
    Matrix _jac, _projection, _directions;
    TangentFunction *_tangent = nullptr;
    bool _hasInput = false, _hasValue = false, _hasJac = false;
};
} // namespace
std::shared_ptr<Function>
EvaluationContext::share(const std::shared_ptr<const Function> &function)
{
    if (!function)
    {
        throw std::invalid_argument("null shared native function");
    }
    auto it = _functions.find(function);
    if (it == _functions.end())
    {
        it = _functions
                 .emplace(function, std::make_shared<CachedFunction>(*function))
                 .first;
    }
    return it->second;
}
void EvaluationContext::invalidate()
{
    for (auto &[source, function] : _functions)
    {
        static_cast<CachedFunction &>(*function).invalidate();
    }
}

void Function::weightedHessian(Input x, Input weights, MatrixOutput hess)
{
    require(x.size() == inputSize() && weights.size() == outputSize() &&
                hess.rows() == inputSize() && hess.cols() == inputSize(),
            "function Hessian size mismatch");
    require(x.allFinite() && weights.allFinite(),
            "function Hessian input is not finite");
    if (!hasHessian())
    {
        throw std::logic_error("function has no exact Hessian");
    }
    weightedHessianImpl(x, weights, hess);
    if (!hess.allFinite())
    {
        throw EvaluationError("function Hessian is not finite");
    }
}

void Function::valueAndJacobian(Input x, VectorOutput value, MatrixOutput jac)
{
    require(x.size() == inputSize() && value.size() == outputSize() &&
                jac.rows() == outputSize() && jac.cols() == inputSize(),
            "function linearization size mismatch");
    require(x.allFinite(), "function input is not finite");
    valueAndJacobianImpl(x, value, jac);
    if (!value.allFinite() || !jac.allFinite())
    {
        throw EvaluationError("function linearization is not finite");
    }
}
void Function::valueAndJacobianImpl(Input x, VectorOutput value,
                                    MatrixOutput jac)
{
    evaluateImpl(x, value);
    jacobianImpl(x, jac);
}

void Function::weightedHessianImpl(Input, Input, MatrixOutput)
{
    throw std::logic_error("function has no exact Hessian");
}

CallbackFunction::CallbackFunction(int nin, int nout, Evaluation value,
                                   Jacobian jac, Hessian hess)
    : Function(nin, nout), _value(std::move(value)), _jac(std::move(jac)),
      _hess(std::move(hess))
{
    require(bool(_value) && bool(_jac),
            "value and Jacobian callbacks are required");
}
std::unique_ptr<Function> CallbackFunction::clone() const
{
    return std::make_unique<CallbackFunction>(*this);
}
void CallbackFunction::evaluateImpl(Input x, VectorOutput v)
{
    _value(x, v);
}
void CallbackFunction::jacobianImpl(Input x, MatrixOutput j)
{
    _jac(x, j);
}
void CallbackFunction::weightedHessianImpl(Input x, Input w, MatrixOutput h)
{
    _hess(x, w, h);
}

ComposedFunction::ComposedFunction(const Function &outer, const Function &inner)
    : Function(inner.inputSize(), outer.outputSize()), _outer(outer.clone()),
      _inner(inner.clone()), _value(inner.outputSize()),
      _weights(inner.outputSize()), _Ji(inner.outputSize(), inner.inputSize()),
      _Jo(outer.outputSize(), outer.inputSize()),
      _Hi(inner.inputSize(), inner.inputSize()),
      _Ho(outer.inputSize(), outer.inputSize()),
      _product(outer.inputSize(), inner.inputSize())
{
    require(outer.inputSize() == inner.outputSize(),
            "composition dimension mismatch");
}
std::unique_ptr<Function> ComposedFunction::clone() const
{
    return std::make_unique<ComposedFunction>(*_outer, *_inner);
}
bool ComposedFunction::hasHessian() const
{
    return _outer->hasHessian() && _inner->hasHessian();
}
void ComposedFunction::evaluateImpl(Input x, VectorOutput v)
{
    _inner->evaluate(x, _value);
    _outer->evaluate(_value, v);
}
void ComposedFunction::setEvaluationContext(
    const std::shared_ptr<EvaluationContext> &context)
{
    _outer->setEvaluationContext(context);
    _inner->setEvaluationContext(context);
}
void ComposedFunction::jacobianImpl(Input x, MatrixOutput j)
{
    _inner->evaluate(x, _value);
    _inner->jacobian(x, _Ji);
    _outer->jacobian(_value, _Jo);
    j.noalias() = _Jo * _Ji;
}
void ComposedFunction::valueAndJacobianImpl(Input x, VectorOutput value,
                                            MatrixOutput jac)
{
    _inner->valueAndJacobian(x, _value, _Ji);
    _outer->valueAndJacobian(_value, value, _Jo);
    jac.noalias() = _Jo * _Ji;
}
void ComposedFunction::weightedHessianImpl(Input x, Input w, MatrixOutput h)
{
    _inner->evaluate(x, _value);
    _inner->jacobian(x, _Ji);
    _outer->jacobian(_value, _Jo);
    _outer->weightedHessian(_value, w, _Ho);
    _weights.noalias() = _Jo.transpose() * w;
    _inner->weightedHessian(x, _weights, _Hi);
    _product.noalias() = _Ho * _Ji;
    h.noalias() = _Ji.transpose() * _product;
    h += _Hi;
}

void FunctionManager::add(const std::string &name,
                          std::shared_ptr<const Function> function,
                          std::vector<Parameter> parameters)
{
    require(!name.empty() && bool(function),
            "function name and implementation are required");
    std::set<std::string> names;
    int size = 0;
    for (const auto &p : parameters)
    {
        require(!p.name.empty() && p.size > 0 && names.insert(p.name).second,
                "parameter names must be unique and sizes positive");
        size += p.size;
    }
    require(size < function->inputSize(), "parameters exhaust function inputs");
    if (!_functions
             .emplace(name,
                      FunctionSpec{std::move(function), std::move(parameters)})
             .second)
    {
        throw std::invalid_argument("function already registered: " + name);
    }
}
const FunctionSpec &FunctionManager::at(const std::string &name) const
{
    return _functions.at(name);
}
} // namespace horizon::ilqr
