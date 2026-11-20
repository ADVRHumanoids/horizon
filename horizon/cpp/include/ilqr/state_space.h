#ifndef HORIZON_ILQR_STATE_SPACE_H
#define HORIZON_ILQR_STATE_SPACE_H

#include "function.h"

namespace horizon::ilqr
{
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

  private:
    std::vector<std::shared_ptr<const StateSpace>> _spaces;
    int _ns = 0, _nt = 0;
};

/** Native maps whose analytical input derivatives use tangent coordinates
 * (e.g. Pinocchio FK with nq stored coordinates and an nv-column Jacobian).
 * Evaluation receives stored coordinates. The adapter handles conversion to
 * the ambient Function interface, including retraction curvature for Hessians.
 * Subclasses implement clone(), evaluateImpl(), and tangentJacobian().
 */
class TangentFunction : public Function
{
  public:
    TangentFunction(std::shared_ptr<const StateSpace> input_space,
                    int output_size);
    const StateSpace &tangentInputSpace() const
    {
        return *_space;
    }
    void valueAndTangentLinearization(Input x, VectorOutput value,
                                      MatrixOutput jac);

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
} // namespace horizon::ilqr
#endif
