#include "ilqr/state_space.h"
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

namespace horizon::ilqr
{
void StateSpace::retractionJacobian(Input x, MatrixOutput jac) const
{
    const double eps = 1e-6;
    Vector d = Vector::Zero(tangentSize()), plus(stateSize()),
           minus(stateSize());
    for (int i = 0; i < tangentSize(); ++i)
    {
        d(i) = eps;
        integrate(x, d, plus);
        integrate(x, -d, minus);
        d(i) = 0;
        jac.col(i) = (plus - minus) / (2 * eps);
    }
}
void StateSpace::tangentProjection(Input x, MatrixOutput projection) const
{
    Matrix R(stateSize(), tangentSize());
    retractionJacobian(x, R);
    projection = (R.transpose() * R).ldlt().solve(R.transpose());
}
void StateSpace::projectDirections(Input x, MatrixInput storage,
                                   MatrixOutput tangent) const
{
    Matrix projection(tangentSize(), stateSize());
    tangentProjection(x, projection);
    tangent.noalias() = projection * storage;
}
void StateSpace::retractionHessian(Input x, Input weights,
                                   MatrixOutput hess) const
{
    const double eps = 1e-4;
    Vector d = Vector::Zero(tangentSize()), e = d;
    Vector pp(stateSize()), pm(stateSize()), mp(stateSize()), mm(stateSize());
    for (int i = 0; i < tangentSize(); ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            d.setZero();
            e.setZero();
            d(i) = eps;
            e(j) = eps;
            integrate(x, d + e, pp);
            integrate(x, d - e, pm);
            integrate(x, -d + e, mp);
            integrate(x, -d - e, mm);
            hess(i, j) = hess(j, i) =
                weights.dot(pp - pm - mp + mm) / (4 * eps * eps);
        }
    }
}
void StateSpace::differenceJacobians(Input from, Input to, MatrixOutput a,
                                     MatrixOutput b) const
{
    const double eps = 1e-6;
    Vector d = Vector::Zero(tangentSize()), plus(stateSize()),
           minus(stateSize());
    Vector gp(tangentSize()), gm(tangentSize());
    for (int i = 0; i < tangentSize(); ++i)
    {
        d(i) = eps;
        integrate(from, d, plus);
        integrate(from, -d, minus);
        difference(plus, to, gp);
        difference(minus, to, gm);
        a.col(i) = (gp - gm) / (2 * eps);
        integrate(to, d, plus);
        integrate(to, -d, minus);
        difference(from, plus, gp);
        difference(from, minus, gm);
        b.col(i) = (gp - gm) / (2 * eps);
        d(i) = 0;
    }
}

CallbackStateSpace::CallbackStateSpace(Vector neutral, int nt, Binary plus,
                                       Binary minus, Jacobian jac, Hessian hess,
                                       DifferenceJacobians diff_jac)
    : _neutral(std::move(neutral)), _nt(nt), _plus(std::move(plus)),
      _minus(std::move(minus)), _jac(std::move(jac)), _hess(std::move(hess)),
      _diff_jac(std::move(diff_jac))
{
    if (_neutral.size() == 0 || !_neutral.allFinite() || nt <= 0 ||
        nt > _neutral.size() || !_plus || !_minus)
    {
        throw std::invalid_argument("invalid custom state space");
    }
}
void CallbackStateSpace::retractionJacobian(Input x, MatrixOutput j) const
{
    if (_jac)
    {
        _jac(x, j);
    }
    else
    {
        StateSpace::retractionJacobian(x, j);
    }
}
void CallbackStateSpace::retractionHessian(Input x, Input w,
                                           MatrixOutput h) const
{
    if (_hess)
    {
        _hess(x, w, h);
    }
    else
    {
        StateSpace::retractionHessian(x, w, h);
    }
}
void CallbackStateSpace::differenceJacobians(Input x, Input y, MatrixOutput a,
                                             MatrixOutput b) const
{
    if (_diff_jac)
    {
        _diff_jac(x, y, a, b);
    }
    else
    {
        StateSpace::differenceJacobians(x, y, a, b);
    }
}

ProductSpace::ProductSpace(
    std::vector<std::shared_ptr<const StateSpace>> spaces)
    : _spaces(std::move(spaces))
{
    if (_spaces.empty())
    {
        throw std::invalid_argument("empty product state space");
    }
    for (const auto &s : _spaces)
    {
        if (!s || s->stateSize() <= 0 || s->tangentSize() <= 0)
        {
            throw std::invalid_argument("invalid product component");
        }
        _ns += s->stateSize();
        _nt += s->tangentSize();
    }
}
void ProductSpace::neutral(VectorOutput x) const
{
    int ns = 0;
    for (const auto &s : _spaces)
    {
        s->neutral(x.segment(ns, s->stateSize()));
        ns += s->stateSize();
    }
}
void ProductSpace::integrate(Input x, Input dx, VectorOutput out) const
{
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->integrate(x.segment(ns, s->stateSize()),
                     dx.segment(nt, s->tangentSize()),
                     out.segment(ns, s->stateSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}
void ProductSpace::difference(Input from, Input to, VectorOutput out) const
{
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->difference(from.segment(ns, s->stateSize()),
                      to.segment(ns, s->stateSize()),
                      out.segment(nt, s->tangentSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}
void ProductSpace::retractionJacobian(Input x, MatrixOutput jac) const
{
    jac.setZero();
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->retractionJacobian(
            x.segment(ns, s->stateSize()),
            jac.block(ns, nt, s->stateSize(), s->tangentSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}
void ProductSpace::tangentProjection(Input x, MatrixOutput projection) const
{
    projection.setZero();
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->tangentProjection(
            x.segment(ns, s->stateSize()),
            projection.block(nt, ns, s->tangentSize(), s->stateSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}
void ProductSpace::projectDirections(Input x, MatrixInput storage,
                                     MatrixOutput tangent) const
{
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->projectDirections(x.segment(ns, s->stateSize()),
                             storage.middleRows(ns, s->stateSize()),
                             tangent.middleRows(nt, s->tangentSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}
void ProductSpace::retractionHessian(Input x, Input w, MatrixOutput hess) const
{
    hess.setZero();
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->retractionHessian(
            x.segment(ns, s->stateSize()), w.segment(ns, s->stateSize()),
            hess.block(nt, nt, s->tangentSize(), s->tangentSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}
void ProductSpace::differenceJacobians(Input from, Input to, MatrixOutput a,
                                       MatrixOutput b) const
{
    a.setZero();
    b.setZero();
    int ns = 0, nt = 0;
    for (const auto &s : _spaces)
    {
        s->differenceJacobians(
            from.segment(ns, s->stateSize()), to.segment(ns, s->stateSize()),
            a.block(nt, nt, s->tangentSize(), s->tangentSize()),
            b.block(nt, nt, s->tangentSize(), s->tangentSize()));
        ns += s->stateSize();
        nt += s->tangentSize();
    }
}

TangentFunction::TangentFunction(std::shared_ptr<const StateSpace> space,
                                 int nout)
    : Function(space ? space->stateSize() : 0, nout), _space(std::move(space))
{
    // Tangent caches provide their own output buffers. Allocate ambient and
    // Hessian scratch lazily, so first-derivative solves do not retain several
    // unused quadratic-size matrices in every node's native clone.
}
void TangentFunction::projection(Input x)
{
    _projection.resize(_space->tangentSize(), inputSize());
    _space->tangentProjection(x, _projection);
}
void TangentFunction::jacobianImpl(Input x, MatrixOutput jac)
{
    _jac.resize(outputSize(), _space->tangentSize());
    projection(x);
    tangentJacobian(x, _jac);
    jac.noalias() = _jac * _projection;
}
void TangentFunction::valueAndTangentJacobian(Input x, VectorOutput value,
                                              MatrixOutput jac)
{
    evaluateImpl(x, value);
    tangentJacobian(x, jac);
}
void TangentFunction::valueAndJacobianImpl(Input x, VectorOutput value,
                                           MatrixOutput jac)
{
    _jac.resize(outputSize(), _space->tangentSize());
    projection(x);
    valueAndTangentJacobian(x, value, _jac);
    jac.noalias() = _jac * _projection;
}
void TangentFunction::valueAndJacobianProductImpl(Input x,
                                                  MatrixInput directions,
                                                  VectorOutput value,
                                                  MatrixOutput product)
{
    _jac.resize(outputSize(), _space->tangentSize());
    valueAndTangentLinearization(x, value, _jac);
    _directions.resize(_space->tangentSize(), directions.cols());
    _space->projectDirections(x, directions, _directions);
    multiplyJacobian(_jac, _directions, product);
}
void TangentFunction::valueAndTangentLinearization(Input x, VectorOutput value,
                                                   MatrixOutput jac)
{
    if (x.size() != inputSize() || value.size() != outputSize() ||
        jac.rows() != outputSize() || jac.cols() != _space->tangentSize())
    {
        throw std::invalid_argument("tangent linearization size mismatch");
    }
    if (!x.allFinite())
    {
        throw std::invalid_argument("tangent input is not finite");
    }
    valueAndTangentJacobian(x, value, jac);
    if (!value.allFinite() || !jac.allFinite())
    {
        throw EvaluationError("tangent linearization is not finite");
    }
}
void TangentFunction::weightedHessianImpl(Input x, Input w, MatrixOutput hess)
{
    const int nt = _space->tangentSize();
    _jac.resize(outputSize(), nt);
    _hess.resize(nt, nt);
    _curvature.resize(nt, nt);
    _gradient.resize(inputSize());
    projection(x);
    tangentJacobian(x, _jac);
    tangentWeightedHessian(x, w, _hess);
    _gradient.noalias() = _projection.transpose() * (_jac.transpose() * w);
    _space->retractionHessian(x, _gradient, _curvature);
    _hess -= _curvature;
    _product.noalias() = _hess * _projection;
    hess.noalias() = _projection.transpose() * _product;
}
void TangentFunction::tangentWeightedHessian(Input, Input, MatrixOutput)
{
    throw std::logic_error("native tangent function has no exact Hessian");
}

namespace
{
Eigen::Matrix3d skew(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d m;
    m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return m;
}
Eigen::Quaterniond quaternion(Input q)
{
    if (q.size() != 4 || !q.allFinite() || std::abs(q.squaredNorm() - 1) > 1e-8)
    {
        throw std::invalid_argument(
            "SO(3) state must be a unit quaternion [x,y,z,w]");
    }
    return Eigen::Quaterniond(q(3), q(0), q(1), q(2));
}
} // namespace
EuclideanSpace::EuclideanSpace(int size) : _size(size)
{
    if (size <= 0)
    {
        throw std::invalid_argument("state size must be positive");
    }
}
void EuclideanSpace::neutral(VectorOutput x) const
{
    x.setZero();
}
void EuclideanSpace::integrate(Input x, Input dx, VectorOutput out) const
{
    out = x + dx;
}
void EuclideanSpace::difference(Input from, Input to, VectorOutput out) const
{
    out = to - from;
}
void EuclideanSpace::retractionJacobian(Input, MatrixOutput j) const
{
    j.setIdentity();
}
void EuclideanSpace::tangentProjection(Input, MatrixOutput p) const
{
    p.setIdentity();
}
void EuclideanSpace::projectDirections(Input, MatrixInput storage,
                                       MatrixOutput tangent) const
{
    tangent = storage;
}
void EuclideanSpace::retractionHessian(Input, Input, MatrixOutput h) const
{
    h.setZero();
}
void EuclideanSpace::differenceJacobians(Input, Input, MatrixOutput a,
                                         MatrixOutput b) const
{
    a.setIdentity();
    a *= -1;
    b.setIdentity();
}

void QuaternionSpace::neutral(VectorOutput x) const
{
    x << 0, 0, 0, 1;
}
void QuaternionSpace::integrate(Input x, Input dx, VectorOutput out) const
{
    const auto q = quaternion(x);
    if (dx.size() != 3 || !dx.allFinite())
    {
        throw std::invalid_argument("SO(3) tangent size must be 3");
    }
    const double theta = dx.norm();
    const double scale =
        theta < 1e-8 ? 0.5 - theta * theta / 48 : std::sin(theta / 2) / theta;
    Eigen::Quaterniond dq(std::cos(theta / 2), scale * dx(0), scale * dx(1),
                          scale * dx(2));
    out = (q * dq).normalized().coeffs();
}
void QuaternionSpace::difference(Input from, Input to, VectorOutput out) const
{
    Eigen::Quaterniond dq = quaternion(from).conjugate() * quaternion(to);
    if (dq.w() < 0)
    {
        dq.coeffs() *= -1;
    }
    const double s = dq.vec().norm();
    out = (s < 1e-8 ? 2.0 : 2 * std::atan2(s, dq.w()) / s) * dq.vec();
}
void QuaternionSpace::retractionJacobian(Input x, MatrixOutput j) const
{
    const auto q = quaternion(x);
    j.topRows(3) = 0.5 * (q.w() * Eigen::Matrix3d::Identity() + skew(q.vec()));
    j.bottomRows(1) = -0.5 * q.vec().transpose();
}
void QuaternionSpace::tangentProjection(Input x, MatrixOutput p) const
{
    Eigen::Matrix<double, 4, 3> R;
    retractionJacobian(x, R);
    p = 4 * R.transpose();
}
void QuaternionSpace::projectDirections(Input x, MatrixInput storage,
                                        MatrixOutput tangent) const
{
    Eigen::Matrix<double, 3, 4> projection;
    tangentProjection(x, projection);
    tangent.noalias() = projection * storage;
}
void QuaternionSpace::retractionHessian(Input x, Input w, MatrixOutput h) const
{
    quaternion(x);
    h.setIdentity();
    h *= -0.25 * x.dot(w);
}
void QuaternionSpace::differenceJacobians(Input from, Input to, MatrixOutput a,
                                          MatrixOutput b) const
{
    Eigen::Vector3d v;
    difference(from, to, v);
    const double theta = v.norm();
    const auto V = skew(v);
    const double c = theta < 1e-5
        ? 1.0 / 12 + theta * theta / 720
        : (1 - 0.5 * theta / std::tan(0.5 * theta)) / (theta * theta);
    const Eigen::Matrix3d common = Eigen::Matrix3d::Identity() + c * V * V;
    a = -common + 0.5 * V;
    b = common + 0.5 * V;
}
} // namespace horizon::ilqr
