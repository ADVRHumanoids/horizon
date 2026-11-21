#ifndef HORIZON_ILQR_CASADI_FUNCTION_H
#define HORIZON_ILQR_CASADI_FUNCTION_H

#include "function.h"
#include "horizon/wrapped_function.h"

namespace horizon::ilqr
{
class StateSpace;

enum class LocalDerivativeKind
{
    Jacobian,
    ExactScalar,
    GaussNewton,
    Dynamics
};

/** Returns a null function when the state space has no symbolic retraction.
 * The outputs are value, local Jacobian and, when requested, local Hessian.
 * For dynamics the Jacobian's output is in the tangent at the prediction.
 */
casadi::Function compileLocalDerivatives(const casadi::Function &function,
                                         const StateSpace &state_space,
                                         LocalDerivativeKind kind);

/** Exposes a pure symbolic stage when a graph contains no native calls. */
class SymbolicStageSource
{
  public:
    virtual ~SymbolicStageSource() = default;
    virtual const casadi::Function &symbolicStage() const = 0;
};

/** CasADi adapter. All column-vector inputs are concatenated in declaration
 * order. One column-vector output is required. AD is constructed once during
 * setup; numerical evaluations reuse WrappedFunction workspaces.
 */
class CasadiFunction final : public Function
{
  public:
    explicit CasadiFunction(const casadi::Function &function,
                            bool exact_hessian = true);
    std::unique_ptr<Function> clone() const override;
    bool hasHessian() const override
    {
        return _exact_hessian;
    }

  protected:
    void evaluateImpl(Input x, VectorOutput value) override;
    void jacobianImpl(Input x, MatrixOutput jac) override;
    void valueAndJacobianImpl(Input x, VectorOutput value,
                              MatrixOutput jac) override;
    void weightedHessianImpl(Input x, Input weights,
                             MatrixOutput hess) override;
    void valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                     VectorOutput value,
                                     MatrixOutput product) override;

  private:
    CasadiFunction(const CasadiFunction &) = default;
    casadi_utils::WrappedFunction _value, _jac, _hess, _linear;
    bool _exact_hessian;
    std::vector<casadi_int> _valueRows, _jacRows, _jacCols;
};

struct SymbolicCall
{
    casadi::MX argument, result;
    std::shared_ptr<const Function> function;
};

/** A symbolic native call is a placeholder until addExpression/addResidual/
 * addCost lowers the expression. Finalize through the same manager rather
 * than constructing a plain CasADi Function containing the placeholder.
 */
class SymbolicNativeFunction
{
  public:
    casadi::MX operator()(const casadi::MX &argument) const;

  private:
    friend class SymbolicFunctionManager;
    std::string _name;
    std::shared_ptr<const Function> _function;
    std::shared_ptr<std::vector<SymbolicCall>> _calls;
};

/** Build ordinary MX expressions around native calls. Pure symbolic segments
 * use CasADi AD; native segments execute directly with their supplied
 * derivatives. Every solver clone owns the complete numerical graph.
 */
class SymbolicFunctionManager : public FunctionManager
{
  public:
    SymbolicNativeFunction addNative(const std::string &name,
                                     std::shared_ptr<const Function> function);
    void addExpression(const std::string &name,
                       const std::vector<casadi::MX> &inputs,
                       const casadi::MX &expression, bool exact_hessian = true);
    void addResidual(const std::string &name,
                     const std::vector<casadi::MX> &inputs,
                     const casadi::MX &expression)
    {
        addExpression(name, inputs, expression, false);
    }
    void addCost(const std::string &name, const std::vector<casadi::MX> &inputs,
                 const casadi::MX &expression)
    {
        addExpression(name, inputs, expression, true);
    }

  private:
    std::shared_ptr<std::vector<SymbolicCall>> _calls =
        std::make_shared<std::vector<SymbolicCall>>();
};
} // namespace horizon::ilqr
#endif
