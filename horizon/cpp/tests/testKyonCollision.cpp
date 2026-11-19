#include <gtest/gtest.h>

#include "horizon/pinocchio_collision.h"
#include <ilqr/state_space.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"

namespace pin = pinocchio;
using namespace horizon::ilqr;
using namespace horizon::collision;

namespace
{
class KyonConfigurationSpace final : public StateSpace
{
public:
    explicit KyonConfigurationSpace(std::shared_ptr<const pin::Model> m) : model(std::move(m)) {}
    int stateSize() const override { return model->nq; }
    int tangentSize() const override { return model->nv; }
    void neutral(VectorOutput q) const override { q = pin::neutral(*model); }
    void integrate(Input q, Input dq, VectorOutput out) const override { out = pin::integrate(*model, q, dq); }
    void difference(Input a, Input b, VectorOutput out) const override { out = pin::difference(*model, a, b); }
    std::shared_ptr<const pin::Model> model;
};
}

TEST(KyonCollision, JointBlockTangentProjectionMatchesDenseReference)
{
    auto model=std::make_shared<pin::Model>();
    pin::urdf::buildModel("data/kyon_spheres.urdf",*model);
    PinocchioConfiguration space(model);
    Matrix actual(model->nv,model->nq),expected(model->nv,model->nq),R(model->nq,model->nv);
    Vector neutral=pin::neutral(*model);
    for(int sample=0;sample<10;++sample)
    {
        Vector delta=Vector::LinSpaced(model->nv,-.2,.3)*(sample+1)/10.;
        Vector q=pin::integrate(*model,neutral,delta);
        // NLP trial points need not have an exactly unit quaternion.
        q.segment<4>(3)*=1.+.001*sample;
        space.tangentProjection(q,actual);
        space.StateSpace::tangentProjection(q,expected);
        space.retractionJacobian(q,R);
        EXPECT_LT((actual-expected).norm(),1e-12);
        EXPECT_LT((actual*R-Matrix::Identity(model->nv,model->nv)).norm(),1e-12);
    }
}

TEST(KyonCollision, SelfCollisionDerivativesAndAvoidance)
{
    auto model = std::make_shared<pin::Model>();
    pin::urdf::buildModel("data/kyon_spheres.urdf", *model);
    auto space = std::make_shared<KyonConfigurationSpace>(model);
    SphereModel spheres(loadSphereModelFromUrdf("data/kyon_spheres.urdf").spheres());
    PinocchioSphereCenters fk(model, space, spheres);
    const int count = spheres.spheres().size();
    Vector q(model->nq), centers(3 * count);
    space->neutral(q);
    Matrix jc(centers.size(), model->nv);
    fk.valueAndTangentLinearization(q, centers, jc);
    // Select a pair on distinct movable branches with a useful separation
    // direction. No automatic exclusions are inferred from a resting pose.
    int first = -1, second = -1;
    double best = 0.;
    for(int i = 0; i < count; ++i) for(int j = i + 1; j < count; ++j)
    {
        if(spheres.spheres()[i].link == spheres.spheres()[j].link) continue;
        Eigen::Vector3d delta;
        Matrix relative(3, model->nv);
        for(int a = 0; a < 3; ++a)
        {
            delta[a] = centers[a * count + i] - centers[a * count + j];
            relative.row(a) = jc.row(a * count + i) - jc.row(a * count + j);
        }
        if(delta.norm() < 1e-6) continue;
        const double score = (delta.normalized().transpose() * relative).norm();
        if(score > best) { best = score; first = i; second = j; }
    }
    ASSERT_GE(first, 0);
    std::vector<double> radii;
    for(const auto& s : spheres.spheres()) radii.push_back(s.radius);
    auto clearance = [&](const Vector& c)
    {
        Eigen::Vector3d d;
        for(int a = 0; a < 3; ++a) d[a] = c[a * count + first] - c[a * count + second];
        return d.norm() - radii[first] - radii[second];
    };
    const double initial_clearance = clearance(centers);
    SphereDistanceFunction collision(radii, {}, {{first, second}}, initial_clearance + .03, .005);
    Vector value(1), plus(1), minus(1);
    Matrix jac(1, centers.size()), product(1, model->nv);
    collision.valueAndJacobian(centers, value, jac);
    collision.valueAndJacobianProduct(centers, jc, plus, product);
    EXPECT_LT((product - jac * jc).norm(), 1e-12);
    for(int col = 0; col < model->nv; ++col)
    {
        Vector step = Vector::Zero(model->nv), qp(model->nq), qm(model->nq), cp(centers.size()), cm(centers.size());
        step[col] = 1e-6;
        space->integrate(q, step, qp); space->integrate(q, -step, qm);
        fk.evaluate(qp, cp); fk.evaluate(qm, cm);
        collision.evaluate(cp, plus); collision.evaluate(cm, minus);
        EXPECT_NEAR(product(0, col), (plus[0] - minus[0]) / 2e-6, 1e-6);
    }
    // Retract descent steps on the actual Kyon configuration manifold.
    for(int iter = 0; iter < 40; ++iter)
    {
        fk.valueAndTangentLinearization(q, centers, jc);
        collision.valueAndJacobianProduct(centers, jc, value, product);
        Vector next(model->nq), step = -0.5 * value[0] * product.row(0).transpose();
        space->integrate(q, step, next); q = next;
    }
    fk.evaluate(q, centers);
    EXPECT_GT(clearance(centers), initial_clearance + .02);
}

TEST(KyonCollision, IlqrSeparatesOverlappingSpheres)
{
    using casadi::MX;
    SymbolicFunctionManager functions;
    auto x = MX::sym("x", 6), u = MX::sym("u", 6);
    auto collision = functions.addNative("self", std::make_shared<SphereDistanceFunction>(
        std::vector<double>{.1, .1}, std::vector<std::shared_ptr<const DistanceField>>{},
        std::vector<SphereDistanceFunction::Pair>{{0, 1}}, .03, .005));
    functions.addExpression("dyn", {x, u}, x + u, false);
    functions.addResidual("avoid", {x, u}, 100 * collision(x));
    functions.addResidual("effort", {x, u}, u);
    Vector start(6); start << -.05, .05, 0, 0, 0, 0;
    horizon::IterativeLQR solver(functions, "dyn", 4);
    Matrix lb = Matrix::Constant(6, 5, -1.), ub = -lb;
    lb.col(0) = ub.col(0) = start;
    solver.setStateBounds(lb, ub);
    solver.setStateInitialGuess(start.replicate(1, 5));
    solver.setInputInitialGuess(Matrix::Zero(6, 4));
    solver.setResidual({1, 2, 3, 4}, functions, "avoid");
    solver.setResidual({0, 1, 2, 3}, functions, "effort");
    solver.solve(30);
    const Vector end = solver.state(4);
    const Eigen::Vector3d d(end[0] - end[1], end[2] - end[3], end[4] - end[5]);
    EXPECT_GT(d.norm(), .22);
}

TEST(KyonCollision, SphereModelFramesAndDistanceJacobian)
{
    auto model = std::make_shared<pin::Model>();
    pin::urdf::buildModel("data/kyon_spheres.urdf", *model);
    auto spheres = std::make_shared<SphereModel>(loadSphereModelFromUrdf("data/kyon_spheres.urdf").spheres());
    ASSERT_EQ(spheres->spheres().size(), 96u);
    for(const auto& sphere : spheres->spheres())
        ASSERT_LT(model->getFrameId(sphere.link), model->frames.size()) << sphere.link;

    auto space = std::make_shared<KyonConfigurationSpace>(model);
    auto workspace = std::make_shared<PinocchioModelWorkspace>(model);
    auto node_context = std::make_shared<EvaluationContext>();
    auto same_node_workspace = workspace->cloneEmpty();
    EXPECT_EQ(workspace->forContext(node_context).get(), same_node_workspace->forContext(node_context).get());
    EXPECT_NE(workspace->forContext(node_context).get(),
              workspace->forContext(std::make_shared<EvaluationContext>()).get());
    PinocchioSphereCenters center_function(model, space, *spheres, workspace);
    center_function.setEvaluationContext(node_context);
    Vector q(model->nq), centers(3 * spheres->spheres().size());
    space->neutral(q);
    center_function.evaluate(q, centers);
    ASSERT_TRUE(centers.allFinite());

    std::vector<double> radii;
    for(const auto& sphere : spheres->spheres()) radii.push_back(sphere.radius);
    auto floor = std::make_shared<PlaneField>(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
    SphereDistanceFunction distance(radii, {floor}, {}, 0.03, 0.01);
    Vector residual(distance.outputSize());
    Matrix dres(distance.outputSize(), centers.size()), dcenter(centers.size(), model->nv);
    distance.valueAndJacobian(centers, residual, dres);
    center_function.valueAndTangentLinearization(q, centers, dcenter);
    Matrix analytic = dres * dcenter;

    const int column = std::min(2, model->nv - 1);
    const double eps = 1e-6;
    Vector delta = Vector::Zero(model->nv), qp(model->nq), qm(model->nq);
    delta[column] = eps;
    space->integrate(q, delta, qp);
    space->integrate(q, -delta, qm);
    Vector cp(centers.size()), cm(centers.size()), rp(residual.size()), rm(residual.size());
    center_function.evaluate(qp, cp); center_function.evaluate(qm, cm);
    distance.evaluate(cp, rp); distance.evaluate(cm, rm);
    EXPECT_LT(((rp - rm) / (2. * eps) - analytic.col(column)).norm(), 2e-4);
}
