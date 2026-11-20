#ifndef HORIZON_ILQR_FUNCTION_H
#define HORIZON_ILQR_FUNCTION_H

#include <Eigen/Core>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

namespace horizon::ilqr
{
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;
using Input = Eigen::Ref<const Vector>;
using VectorOutput = Eigen::Ref<Vector>;
using MatrixOutput = Eigen::Ref<Matrix>;
using MatrixInput = Eigen::Ref<const Matrix>;
class EvaluationContext;

/** Numerical failure at an otherwise correctly sized input. Line search may
 * retry a shorter step; failures at the accepted iterate still propagate. */
struct EvaluationError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/** A differentiable map R^n -> R^m, with an independently owned workspace.
 *
 * Derivatives use ambient input coordinates. weightedHessian computes
 * sum_i weights[i] * Hessian(f_i), which avoids a third-order tensor.
 * clone() must duplicate mutable workspace, including third-party library data.
 * Calls on one instance are sequential; different clones may run concurrently.
 */
class Function
{
  public:
    Function(int input_size, int output_size);
    virtual ~Function() = default;
    int inputSize() const
    {
        return _input_size;
    }
    int outputSize() const
    {
        return _output_size;
    }
    virtual std::unique_ptr<Function> clone() const = 0;
    virtual bool hasHessian() const
    {
        return false;
    }
    /** Prefer the implementation's directional kernel over caching a dense
     * Jacobian. Useful when that Jacobian is large and structurally sparse. */
    virtual bool hasJacobianProduct() const
    {
        return false;
    }
    virtual void
    setEvaluationContext(const std::shared_ptr<EvaluationContext> &)
    {
    }

    void evaluate(Input x, VectorOutput value);
    void jacobian(Input x, MatrixOutput jac);
    void valueAndJacobian(Input x, VectorOutput value, MatrixOutput jac);
    void weightedHessian(Input x, Input weights, MatrixOutput hess);
    /** Evaluate and propagate arbitrary input directions, e.g. [Rx, Iu, 0p]. */
    void valueAndJacobianProduct(Input x, MatrixInput directions,
                                 VectorOutput value, MatrixOutput product);

  protected:
    static void multiplyJacobian(MatrixInput jac, MatrixInput directions,
                                 MatrixOutput product);
    virtual void evaluateImpl(Input x, VectorOutput value) = 0;
    virtual void jacobianImpl(Input x, MatrixOutput jac) = 0;
    virtual void valueAndJacobianImpl(Input x, VectorOutput value,
                                      MatrixOutput jac);
    virtual void weightedHessianImpl(Input x, Input weights, MatrixOutput hess);
    virtual void valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                             VectorOutput value,
                                             MatrixOutput product);

  private:
    int _input_size, _output_size;
    Matrix _productJacobian;
};

/** Node-local memoization for explicitly registered symbolic native calls.
 * One context is used sequentially; different nodes/solvers own different
 * contexts. Inputs are compared bitwise and failed evaluations are not cached.
 */
class EvaluationContext
{
  public:
    std::shared_ptr<Function>
    share(const std::shared_ptr<const Function> &function);
    /** Share an arbitrary node-local backend workspace between native
     * functions. The key must be the common immutable prototype identity.
     */
    template <class T, class Factory>
    std::shared_ptr<T> shareObject(const void *key, Factory &&factory)
    {
        if (!key)
        {
            throw std::invalid_argument(
                "evaluation-context object key is null");
        }
        const auto typed_key = std::make_pair(key, std::type_index(typeid(T)));
        auto it = _objects.find(typed_key);
        if (it != _objects.end())
        {
            return std::static_pointer_cast<T>(it->second);
        }
        auto value = std::forward<Factory>(factory)();
        if (!value)
        {
            throw std::invalid_argument(
                "evaluation-context factory returned null");
        }
        auto typed = std::static_pointer_cast<T>(value);
        _objects.emplace(typed_key, std::move(value));
        return typed;
    }
    void invalidate();

  private:
    struct IdentityLess
    {
        bool operator()(const std::shared_ptr<const Function> &a,
                        const std::shared_ptr<const Function> &b) const
        {
            return std::less<const Function *>{}(a.get(), b.get());
        }
    };
    std::map<std::shared_ptr<const Function>, std::shared_ptr<Function>,
             IdentityLess>
        _functions;
    std::map<std::pair<const void *, std::type_index>, std::shared_ptr<void>>
        _objects;
};

/** Prepare shared numerical work for all shooting nodes before independent
 * node evaluation. Called on the solver thread, including every trial cost
 * pass. Implementations must own their buffers and publish results through
 * the supplied node contexts; clones keep independent mutable state. */
class EvaluationBatch
{
  public:
    virtual ~EvaluationBatch() = default;
    virtual std::unique_ptr<EvaluationBatch> clone() const = 0;
    virtual void
    prepare(MatrixInput states, MatrixInput controls,
            const std::vector<std::shared_ptr<EvaluationContext>> &contexts,
            bool derivatives) = 0;
};

/** State-only horizon cost, already differentiated in tangent coordinates.
 * Results are unscaled: iLQR applies its running/terminal normalization.
 * Hessians are Gauss-Newton approximations, packed by rows of the upper
 * triangle: (0,0),(0,1),...,(1,1),... . Failed prepares invalidate all results.
 */
class BatchedCost
{
  public:
    virtual ~BatchedCost() = default;
    virtual std::unique_ptr<BatchedCost> clone() const = 0;
    virtual int stateSize() const = 0;
    virtual int tangentSize() const = 0;
    virtual void prepare(MatrixInput states, bool derivatives) = 0;
    virtual double value(int node) const = 0;
    virtual Input gradient(int node) const = 0;
    virtual Input packedHessian(int node) const = 0;
};

/** Convenience adapter for stateless/reentrant C++ callbacks. For mutable
 * Pinocchio Data or similar workspace, implement Function::clone() instead.
 */
class CallbackFunction final : public Function
{
  public:
    using Evaluation = std::function<void(Input, VectorOutput)>;
    using Jacobian = std::function<void(Input, MatrixOutput)>;
    using Hessian = std::function<void(Input, Input, MatrixOutput)>;
    CallbackFunction(int nin, int nout, Evaluation value, Jacobian jac,
                     Hessian hess = {});
    std::unique_ptr<Function> clone() const override;
    bool hasHessian() const override
    {
        return bool(_hess);
    }

  protected:
    void evaluateImpl(Input x, VectorOutput value) override;
    void jacobianImpl(Input x, MatrixOutput jac) override;
    void weightedHessianImpl(Input x, Input weights,
                             MatrixOutput hess) override;

  private:
    Evaluation _value;
    Jacobian _jac;
    Hessian _hess;
};

/** outer(inner(x)); both Jacobians and exact second derivatives are composed.
 */
class ComposedFunction final : public Function
{
  public:
    ComposedFunction(const Function &outer, const Function &inner);
    std::unique_ptr<Function> clone() const override;
    bool hasHessian() const override;
    void setEvaluationContext(
        const std::shared_ptr<EvaluationContext> &context) override;

  protected:
    void evaluateImpl(Input x, VectorOutput value) override;
    void jacobianImpl(Input x, MatrixOutput jac) override;
    void valueAndJacobianImpl(Input x, VectorOutput value,
                              MatrixOutput jac) override;
    void weightedHessianImpl(Input x, Input weights,
                             MatrixOutput hess) override;

  private:
    std::unique_ptr<Function> _outer, _inner;
    Vector _value, _weights;
    Matrix _Ji, _Jo, _Hi, _Ho, _product;
};

struct Parameter
{
    std::string name;
    int size;
};

/** A stage map receives [state; control; parameters...] in that order. */
struct FunctionSpec
{
    std::shared_ptr<const Function> function;
    std::vector<Parameter> parameters;
};

/** Registry is external to the solver. Registration is a setup operation;
 * solver workspaces are cloned, so later registry edits do not affect a solve.
 */
class FunctionManager
{
  public:
    void add(const std::string &name, std::shared_ptr<const Function> function,
             std::vector<Parameter> parameters = {});
    const FunctionSpec &at(const std::string &name) const;

  private:
    std::map<std::string, FunctionSpec> _functions;
};
} // namespace horizon::ilqr
#endif
