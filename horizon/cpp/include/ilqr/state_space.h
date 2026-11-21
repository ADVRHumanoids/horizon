#ifndef HORIZON_ILQR_STATE_SPACE_H
#define HORIZON_ILQR_STATE_SPACE_H

#include "function.h"

namespace horizon::ilqr
{
struct DefectBlock
{
    int offset, size;
    bool euclidean;
};

/** Retraction and local difference. State storage and tangent dimensions may
 * differ (e.g. quaternion SO(3): 4 and 3). Implementations must be reentrant.
 * difference(from, to) is the tangent at from taking from to to.
 */
class StateSpace
{
  public:
    virtual ~StateSpace() = default;
    virtual int stateSize() const = 0;
    virtual int tangentSize() const = 0;
    virtual void neutral(VectorOutput x) const = 0;
    virtual void integrate(Input x, Input dx, VectorOutput result) const = 0;
    virtual void difference(Input from, Input to,
                            VectorOutput result) const = 0;
    /** d integrate(x, dx) / d dx at dx=0, in storage coordinates. */
    virtual void retractionJacobian(Input x, MatrixOutput jac) const;
    /** Moore-Penrose left inverse of the retraction Jacobian. */
    virtual void tangentProjection(Input x, MatrixOutput projection) const;
    virtual void projectDirections(Input x, MatrixInput storage_directions,
                                   MatrixOutput tangent_directions) const;
    /** Hessian of weights.dot(integrate(x, dx)) at dx=0. */
    virtual void retractionHessian(Input x, Input weights,
                                   MatrixOutput hess) const;
    /** Derivatives of difference(integrate(from,a), integrate(to,b)) at a=b=0.
     */
    virtual void differenceJacobians(Input from, Input to, MatrixOutput dfrom,
                                     MatrixOutput dto) const;
    /** Independent diagonal blocks of the difference Jacobians. Offsets are
     * in this space's tangent coordinates. The default is one coupled block.
     */
    virtual void defectBlocks(std::vector<DefectBlock> &blocks) const;
    /** Difference, its two Jacobians, transport from the prediction tangent
     * to the shooting-state tangent, and the transported affine defect.
     */
    void linearizeDefect(Input from, Input to, VectorOutput gap,
                         MatrixOutput dfrom, MatrixOutput dto,
                         MatrixOutput transport, VectorOutput defect) const;
};

class EuclideanSpace final : public StateSpace
{
  public:
    explicit EuclideanSpace(int size);
    int stateSize() const override
    {
        return _size;
    }
    int tangentSize() const override
    {
        return _size;
    }
    void neutral(VectorOutput x) const override;
    void integrate(Input x, Input dx, VectorOutput result) const override;
    void difference(Input from, Input to, VectorOutput result) const override;
    void retractionJacobian(Input x, MatrixOutput jac) const override;
    void tangentProjection(Input x, MatrixOutput projection) const override;
    void projectDirections(Input x, MatrixInput storage_directions,
                           MatrixOutput tangent_directions) const override;
    void retractionHessian(Input x, Input weights,
                           MatrixOutput hess) const override;
    void differenceJacobians(Input from, Input to, MatrixOutput a,
                             MatrixOutput b) const override;
    void defectBlocks(std::vector<DefectBlock> &blocks) const override;

  private:
    int _size;
};

/** Unit quaternion SO(3), stored [qx,qy,qz,qw], with right/body increments.
 * Differences choose the shortest rotation; the pi branch cut is nonsmooth.
 */
class QuaternionSpace final : public StateSpace
{
  public:
    int stateSize() const override
    {
        return 4;
    }
    int tangentSize() const override
    {
        return 3;
    }
    void neutral(VectorOutput x) const override;
    void integrate(Input x, Input dx, VectorOutput result) const override;
    void difference(Input from, Input to, VectorOutput result) const override;
    void retractionJacobian(Input x, MatrixOutput jac) const override;
    void tangentProjection(Input x, MatrixOutput projection) const override;
    void projectDirections(Input x, MatrixInput storage_directions,
                           MatrixOutput tangent_directions) const override;
    void retractionHessian(Input x, Input weights,
                           MatrixOutput hess) const override;
    void differenceJacobians(Input from, Input to, MatrixOutput a,
                             MatrixOutput b) const override;
};
/** Arbitrary user-supplied oplus/ominus. Derivatives default to central finite
 * differences; supply analytical callbacks for performance-sensitive solves.
 * Operations and any captured third-party workspace must be reentrant.
 */
class CallbackStateSpace final : public StateSpace
{
  public:
    using Binary = std::function<void(Input, Input, VectorOutput)>;
    using Jacobian = std::function<void(Input, MatrixOutput)>;
    using Hessian = std::function<void(Input, Input, MatrixOutput)>;
    using DifferenceJacobians =
        std::function<void(Input, Input, MatrixOutput, MatrixOutput)>;
    CallbackStateSpace(Vector neutral, int tangent_size, Binary oplus,
                       Binary ominus, Jacobian jac = {}, Hessian hess = {},
                       DifferenceJacobians diff_jac = {});
    int stateSize() const override
    {
        return _neutral.size();
    }
    int tangentSize() const override
    {
        return _nt;
    }
    void neutral(VectorOutput x) const override
    {
        x = _neutral;
    }
    void integrate(Input x, Input dx, VectorOutput out) const override
    {
        _plus(x, dx, out);
    }
    void difference(Input from, Input to, VectorOutput out) const override
    {
        _minus(from, to, out);
    }
    void retractionJacobian(Input x, MatrixOutput jac) const override;
    void retractionHessian(Input x, Input w, MatrixOutput hess) const override;
    void differenceJacobians(Input from, Input to, MatrixOutput a,
                             MatrixOutput b) const override;

  private:
    Vector _neutral;
    int _nt;
    Binary _plus, _minus;
    Jacobian _jac;
    Hessian _hess;
    DifferenceJacobians _diff_jac;
};

/** Cartesian product, useful for [robot configuration; velocity; control; p].
 */
class ProductSpace final : public StateSpace
{
  public:
    explicit ProductSpace(
        std::vector<std::shared_ptr<const StateSpace>> spaces);
    int stateSize() const override
    {
        return _ns;
    }
    int tangentSize() const override
    {
        return _nt;
    }
    const std::vector<std::shared_ptr<const StateSpace>> &components() const
    {
        return _spaces;
    }
    void neutral(VectorOutput x) const override;
    void integrate(Input x, Input dx, VectorOutput out) const override;
    void difference(Input from, Input to, VectorOutput out) const override;
    void retractionJacobian(Input x, MatrixOutput jac) const override;
    void tangentProjection(Input x, MatrixOutput projection) const override;
    void projectDirections(Input x, MatrixInput storage_directions,
                           MatrixOutput tangent_directions) const override;
    void retractionHessian(Input x, Input w, MatrixOutput hess) const override;
    void differenceJacobians(Input from, Input to, MatrixOutput a,
                             MatrixOutput b) const override;
    void defectBlocks(std::vector<DefectBlock> &blocks) const override;

  private:
    std::vector<std::shared_ptr<const StateSpace>> _spaces;
    int _ns = 0, _nt = 0;
};

/** A native map whose Jacobian columns use its input space's tangent chart. */
class TangentLinearization
{
  public:
    virtual ~TangentLinearization() = default;
    virtual const StateSpace *localInputSpace() const = 0;
    virtual void valueAndTangentLinearization(Input x, VectorOutput value,
                                              MatrixOutput jac) = 0;
    virtual bool hasTangentHessian() const
    {
        return false;
    }
    virtual void weightedTangentHessian(Input, Input, MatrixOutput)
    {
        throw std::logic_error("tangent Hessian unavailable");
    }
};

/** Stage linearization for canonical [dx; du] directions, with parameters
 * fixed. Retraction is the storage derivative at the state in input.
 */
class StageTangentLinearization
{
  public:
    virtual ~StageTangentLinearization() = default;
    virtual void
    valueAndStageLinearization(Input input, const StateSpace &state_space,
                               int control_size, MatrixInput retraction,
                               VectorOutput value, MatrixOutput jac) = 0;
};

class StageTangentQuadratic
{
  public:
    virtual ~StageTangentQuadratic() = default;
    virtual void valueAndStageQuadratic(Input input, const StateSpace &space,
                                        int control_size,
                                        MatrixInput retraction,
                                        VectorOutput value,
                                        VectorOutput gradient,
                                        MatrixOutput hessian) = 0;
};

/** Native maps whose analytical input derivatives use tangent coordinates
 * (e.g. Pinocchio FK with nq stored coordinates and an nv-column Jacobian).
 * Evaluation receives stored coordinates. The adapter handles conversion to
 * the ambient Function interface, including retraction curvature for Hessians.
 * Subclasses implement clone(), evaluateImpl(), and tangentJacobian().
 */
class TangentFunction : public Function, public TangentLinearization
{
  public:
    TangentFunction(std::shared_ptr<const StateSpace> input_space,
                    int output_size);
    const StateSpace &tangentInputSpace() const
    {
        return *_space;
    }
    const StateSpace *localInputSpace() const override
    {
        return _space.get();
    }
    void valueAndTangentLinearization(Input x, VectorOutput value,
                                      MatrixOutput jac) override;
    bool hasTangentHessian() const override
    {
        return hasHessian();
    }
    void weightedTangentHessian(Input x, Input weights,
                                MatrixOutput hess) override;

  protected:
    virtual void tangentJacobian(Input x, MatrixOutput jac) = 0;
    virtual void valueAndTangentJacobian(Input x, VectorOutput value,
                                         MatrixOutput jac);
    virtual void tangentWeightedHessian(Input x, Input w, MatrixOutput hess);

  private:
    void jacobianImpl(Input x, MatrixOutput jac) final;
    void valueAndJacobianImpl(Input x, VectorOutput value,
                              MatrixOutput jac) final;
    void weightedHessianImpl(Input x, Input w, MatrixOutput hess) final;
    void valueAndJacobianProductImpl(Input x, MatrixInput directions,
                                     VectorOutput value,
                                     MatrixOutput product) final;
    void projection(Input x);
    std::shared_ptr<const StateSpace> _space;
    Matrix _R, _projection, _jac, _hess, _curvature, _product, _directions;
    Vector _gradient;
};

/** A stage dynamics map with input [state; Euclidean controls/parameters].
 * Its output is stored in state coordinates; its Jacobian maps input tangent
 * directions to the tangent at the returned prediction. The exact same state
 * space object must be used by the solver before this contract can be used.
 */
class TangentDynamics : public TangentFunction
{
  public:
    TangentDynamics(std::shared_ptr<const StateSpace> state_space,
                    int euclidean_inputs);
    const StateSpace &stateSpace() const
    {
        return *_state_space;
    }
    void valueAndTangentOutputLinearization(Input input, VectorOutput value,
                                            MatrixOutput jacobian);

  protected:
    virtual void tangentOutputLinearization(Input input, VectorOutput value,
                                            MatrixOutput jacobian) = 0;

  private:
    void tangentJacobian(Input input, MatrixOutput jacobian) final;
    void valueAndTangentJacobian(Input input, VectorOutput value,
                                 MatrixOutput jacobian) final;
    std::shared_ptr<const StateSpace> _state_space;
    Matrix _local_jacobian, _output_retraction;
    Vector _prediction;
};
} // namespace horizon::ilqr
#endif
