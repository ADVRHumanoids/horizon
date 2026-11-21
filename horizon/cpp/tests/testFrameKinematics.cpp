#include "horizon/pinocchio_kinematics.h"
#include "horizon/pinocchio_robot.h"
#include "ilqr/ilqr_impl.h"
#include <Eigen/Geometry>
#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/spatial/explog.hpp>

using namespace horizon::ilqr;
using namespace horizon::kinematics;
namespace pin = pinocchio;

namespace horizon
{
struct IterativeLQRTestAccess
{
    using Dynamics = IterativeLQR::Dynamics;
    using FunctionWorkspace = IterativeLQR::FunctionWorkspace;
};
} // namespace horizon

namespace
{
struct Robot
{
    std::shared_ptr<pin::Model> model = std::make_shared<pin::Model>();
    std::shared_ptr<PinocchioConfiguration> space;
    Vector q;
    pin::FrameIndex tool, contact;
    Robot()
    {
        auto base = model->addJoint(0, pin::JointModelFreeFlyer(),
                                    pin::SE3::Identity(), "base");
        auto arm = model->addJoint(base, pin::JointModelRY(),
                                   pin::SE3::Identity(), "arm");
        tool = model->addFrame(
            pin::Frame("tool", arm, 0,
                       pin::SE3(Eigen::AngleAxisd(.3, Eigen::Vector3d::UnitX())
                                    .toRotationMatrix(),
                                Eigen::Vector3d(.7, -.1, .2)),
                       pin::OP_FRAME));
        contact =
            model->addFrame(pin::Frame("contact", base, 0,
                                       pin::SE3(Eigen::Matrix3d::Identity(),
                                                Eigen::Vector3d(-.3, .2, -.4)),
                                       pin::OP_FRAME));
        space = std::make_shared<PinocchioConfiguration>(model);
        q = pin::integrate(*model, pin::neutral(*model),
                           Vector::LinSpaced(model->nv, -.2, .3));
    }
};

Matrix numericalTangent(TangentFunction &function, const Vector &z,
                        double eps = 1e-7)
{
    const auto &space = function.tangentInputSpace();
    Matrix jac(function.outputSize(), space.tangentSize());
    Vector delta = Vector::Zero(space.tangentSize()), plus(z.size()),
           minus(z.size());
    Vector a(function.outputSize()), b(function.outputSize());
    for (int i = 0; i < delta.size(); ++i)
    {
        delta[i] = eps;
        space.integrate(z, delta, plus);
        delta[i] = -eps;
        space.integrate(z, delta, minus);
        delta[i] = 0;
        function.evaluate(plus, a);
        function.evaluate(minus, b);
        jac.col(i) = (a - b) / (2 * eps);
    }
    return jac;
}
} // namespace

TEST(FrameKinematics, TangentOutputStepAndInfeasibleDefectTransport)
{
    Robot robot;
    const int nq = robot.model->nq, nv = robot.model->nv;
    auto step = std::make_shared<PinocchioStep>(robot.space, .13);
    Vector z(nq + nv), prediction(nq), plus(nq + nv), minus(nq + nv), yp(nq),
        ym(nq), dp(nv), dm(nv);
    z << robot.q, Vector::LinSpaced(nv, -.4, .3);
    Matrix local(nv, 2 * nv), numeric(nv, 2 * nv);
    step->valueAndTangentOutputLinearization(z, prediction, local);
    for (int i = 0; i < 2 * nv; ++i)
    {
        Vector direction = Vector::Zero(2 * nv);
        direction(i) = 1e-7;
        step->tangentInputSpace().integrate(z, direction, plus);
        step->tangentInputSpace().integrate(z, -direction, minus);
        step->evaluate(plus, yp);
        step->evaluate(minus, ym);
        robot.space->difference(prediction, yp, dp);
        robot.space->difference(prediction, ym, dm);
        numeric.col(i) = (dp - dm) / 2e-7;
    }
    EXPECT_LT((local - numeric).norm(), 1e-7);

    using Access = horizon::IterativeLQRTestAccess;
    FunctionSpec spec{step, {}};
    Access::Dynamics direct(nq, nv), fallback(nq, nv);
    direct.space = robot.space;
    fallback.space = std::make_shared<PinocchioConfiguration>(robot.model);
    direct.native = std::make_shared<Access::FunctionWorkspace>(spec, nq, nv);
    fallback.native = std::make_shared<Access::FunctionWorkspace>(spec, nq, nv);
    for (double offset : {0., .3, -1.2})
    {
        Vector next(nq), delta = Vector::Constant(nv, offset);
        robot.space->integrate(prediction, delta, next);
        for (double sign : {1., -1.})
        {
            next.segment<4>(3) *= sign;
            direct.linearize(z.head(nq), z.tail(nv), next, 0);
            fallback.linearize(z.head(nq), z.tail(nv), next, 0);
            EXPECT_LT((direct.A() - fallback.A()).norm(), 1e-10);
            EXPECT_LT((direct.B() - fallback.B()).norm(), 1e-10);
            EXPECT_LT((direct.d - fallback.d).norm(), 1e-10);
            Vector gap(nv), defect(nv);
            Matrix A(nv, nv), B(nv, nv), transport(nv, nv);
            robot.space->linearizeDefect(next, prediction, gap, A, B, transport,
                                         defect);
            Eigen::FullPivLU<Matrix> dense(A);
            ASSERT_TRUE(dense.isInvertible());
            EXPECT_LT((transport + dense.solve(B)).norm(), 1e-10);
            EXPECT_LT((defect + dense.solve(gap)).norm(), 1e-10);
            next.segment<4>(3) *= sign;
        }
    }
    Matrix wrong = Matrix::Zero(nv, nv);
    EXPECT_THROW(step->valueAndTangentOutputLinearization(z, prediction, wrong),
                 std::invalid_argument);
}

TEST(FrameKinematics, DISABLED_ProfileTangentOutputDefectLinearization)
{
    Robot robot;
    const int nq = robot.model->nq, nv = robot.model->nv;
    auto step = std::make_shared<PinocchioStep>(robot.space, .13);
    FunctionSpec spec{step, {}};
    using Access = horizon::IterativeLQRTestAccess;
    Access::Dynamics direct(nq, nv), fallback(nq, nv);
    direct.space = robot.space;
    fallback.space = std::make_shared<PinocchioConfiguration>(robot.model);
    direct.native = std::make_shared<Access::FunctionWorkspace>(spec, nq, nv);
    fallback.native = std::make_shared<Access::FunctionWorkspace>(spec, nq, nv);
    Vector u = Vector::LinSpaced(nv, -.4, .3), next(nq);
    robot.space->integrate(robot.q, Vector::Constant(nv, .2), next);
    double checksum = 0;
    auto run = [&](Access::Dynamics &d)
    {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10000; ++i)
        {
            d.linearize(robot.q, u, next, 0);
            checksum += d.A()(0, 0);
        }
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - start)
                   .count() /
            10000;
    };
    if (const char *mode = std::getenv("HORIZON_BENCH_DYNAMICS_PATH"))
    {
        const std::string path(mode);
        ASSERT_TRUE(path == "direct" || path == "fallback");
        const double elapsed = run(path == "direct" ? direct : fallback);
        std::cout << path
                  << " tangent defect linearization us/call: " << elapsed
                  << '\n';
        EXPECT_TRUE(std::isfinite(checksum));
        return;
    }
    run(direct);
    run(fallback);
    for (int i = 0; i < 3; ++i)
    {
        const double a = run(direct), b = run(fallback);
        std::cout << "tangent defect linearization us/call: direct=" << a
                  << " fallback=" << b << '\n';
    }
    EXPECT_TRUE(std::isfinite(checksum));
}

TEST(FrameKinematics, DISABLED_ProfileStructuredDefectTransport)
{
    for (int joints : {1, 12, 30})
    {
        auto model = std::make_shared<pin::Model>();
        auto base = model->addJoint(0, pin::JointModelFreeFlyer(),
                                    pin::SE3::Identity(), "base");
        for (int i = 0; i < joints; ++i)
        {
            model->addJoint(base, pin::JointModelRY(), pin::SE3::Identity(),
                            "joint" + std::to_string(i));
        }
        PinocchioConfiguration space(model);
        const int nt = space.tangentSize();
        Vector from(space.stateSize()), to(space.stateSize()), gap(nt),
            defect(nt), delta = Vector::LinSpaced(nt, -.2, .3);
        space.neutral(from);
        space.integrate(from, delta, to);
        Matrix A(nt, nt), B(nt, nt), transport(nt, nt);
        double checksum = 0;
        auto run = [&](bool structured)
        {
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10000; ++i)
            {
                if (structured)
                {
                    space.linearizeDefect(from, to, gap, A, B, transport,
                                          defect);
                }
                else
                {
                    space.difference(from, to, gap);
                    space.differenceJacobians(from, to, A, B);
                    Eigen::FullPivLU<Matrix> lu(A);
                    transport = -lu.solve(B);
                    defect = -lu.solve(gap);
                }
                checksum += transport(0, 0) + defect(0);
            }
            return std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - start)
                       .count() /
                10000;
        };
        run(true);
        run(false);
        for (int i = 0; i < 3; ++i)
        {
            const double structured = run(true), dense = run(false);
            std::cout << "defect transport nv=" << nt
                      << " us/call: structured=" << structured
                      << " dense=" << dense << '\n';
        }
        EXPECT_TRUE(std::isfinite(checksum));
    }
}

TEST(FrameKinematics, PositionsAndPoseMatchIndependentPinocchioAndDerivatives)
{
    Robot robot;
    pin::Data reference(*robot.model);
    pin::forwardKinematics(*robot.model, reference, robot.q);
    pin::updateFramePlacements(*robot.model, reference);
    PinocchioFramePositions positions(robot.model, robot.space,
                                      {"contact", "tool"});
    Vector value(6);
    Matrix jac(6, robot.model->nv);
    positions.valueAndTangentLinearization(robot.q, value, jac);
    for (int i = 0; i < 2; ++i)
    {
        const auto &p =
            reference.oMf[i == 0 ? robot.contact : robot.tool].translation();
        EXPECT_NEAR(value[i], p.x(), 1e-13);
        EXPECT_NEAR(value[2 + i], p.y(), 1e-13);
        EXPECT_NEAR(value[4 + i], p.z(), 1e-13);
    }
    EXPECT_LT((jac - numericalTangent(positions, robot.q)).norm(), 2e-8);
    PinocchioFramePose pose(robot.model, robot.space, "tool");
    Vector actual(7);
    Matrix pose_jac(7, robot.model->nv);
    pose.valueAndTangentLinearization(robot.q, actual, pose_jac);
    EXPECT_LT(
        (actual.head<3>() - reference.oMf[robot.tool].translation()).norm(),
        1e-13);
    Eigen::Quaterniond orientation;
    orientation.coeffs() = actual.tail<4>();
    EXPECT_LT(
        (orientation.toRotationMatrix() - reference.oMf[robot.tool].rotation())
            .norm(),
        1e-13);
    EXPECT_LT((pose_jac - numericalTangent(pose, robot.q)).norm(), 2e-8);
}

TEST(FrameKinematics, PoseErrorConfigurationAndTargetDerivativesZeroAndNearPi)
{
    Robot robot;
    PinocchioFramePose pose(robot.model, robot.space, "tool");
    PinocchioFramePoseError error(robot.model, robot.space, "tool");
    Vector target(7);
    pose.evaluate(robot.q, target);
    const Eigen::Quaterniond actual_orientation(target[6], target[3], target[4],
                                                target[5]);
    for (double angle : {0., .7, 3.141592653589793 - .01})
    {
        Vector z(robot.model->nq + 7);
        z << robot.q, target;
        const Eigen::Quaterniond rotation(
            Eigen::AngleAxisd(angle, Eigen::Vector3d(1, 2, 3).normalized()));
        Eigen::Quaterniond desired = actual_orientation * rotation;
        z.tail<4>() = 1.3 * desired.coeffs();
        z.segment<3>(robot.model->nq) += Eigen::Vector3d(.02, -.03, .01);
        Vector value(6);
        Matrix jac(6, robot.model->nv + 7);
        error.valueAndTangentLinearization(z, value, jac);
        EXPECT_LT((value.head<3>() - Eigen::Vector3d(-.02, .03, -.01)).norm(),
                  1e-12);
        EXPECT_NEAR(value.tail<3>().norm(), angle, 1e-10);
        EXPECT_LT(
            (jac - numericalTangent(error, z, angle > 3 ? 1e-5 : 1e-7)).norm(),
            angle > 3 ? 5e-6 : 3e-8);
        Vector opposite = z, other(6);
        opposite.tail<4>() *= -1;
        error.evaluate(opposite, other);
        EXPECT_LT((value - other).norm(), 1e-12);
        Matrix ambient(6, z.size()), retraction(z.size(), robot.model->nv + 7);
        error.jacobian(z, ambient);
        error.tangentInputSpace().retractionJacobian(z, retraction);
        EXPECT_LT((ambient * retraction - jac).norm(), 1e-11);
    }
    Vector z(robot.model->nq + 7), zero(6);
    z << robot.q, target;
    error.evaluate(z, zero);
    EXPECT_LT(zero.norm(), 1e-13);
}

TEST(FrameKinematics, AmbientDerivativesAtNonunitTrialConfigurations)
{
    Robot robot;
    PinocchioFramePositions positions(robot.model, robot.space,
                                      {"contact", "tool"});
    PinocchioFramePose pose(robot.model, robot.space, "tool");
    PinocchioFramePoseError error(robot.model, robot.space, "tool");
    PinocchioStep step(robot.space, .2);
    Vector target(7);
    pose.evaluate(robot.q, target);
    target.head<3>() += Eigen::Vector3d(.01, .02, -.01);
    target.tail<4>() *= 1.2;
    Vector q = robot.q;
    q.segment<4>(3) *= 1.05;
    Vector pose_input(q.size() + 7);
    pose_input << q, target;
    Vector step_input(q.size() + robot.model->nv);
    step_input << q, Vector::LinSpaced(robot.model->nv, -.1, .2);
    for (Function *function :
         std::vector<Function *>{&positions, &pose, &error, &step})
    {
        Vector input = function == &error
            ? pose_input
            : (function == &step ? step_input : q);
        Matrix jac(function->outputSize(), input.size()),
            numerical(jac.rows(), jac.cols());
        function->jacobian(input, jac);
        for (int k = 0; k < input.size(); ++k)
        {
            Vector plus = input, minus = input;
            Vector a(function->outputSize()), b(function->outputSize());
            plus[k] += 1e-7;
            minus[k] -= 1e-7;
            function->evaluate(plus, a);
            function->evaluate(minus, b);
            numerical.col(k) = (a - b) / (2e-7);
        }
        EXPECT_LT((jac - numerical).norm(), 3e-8);
    }
}

TEST(FrameKinematics, CompositeAndPlanarTrialCoordinates)
{
    auto model = std::make_shared<pin::Model>();
    pin::JointModelComposite composite;
    composite.addJoint(pin::JointModelFreeFlyer());
    composite.addJoint(pin::JointModelRY());
    auto root = model->addJoint(0, composite, pin::SE3::Identity(), "root");
    auto planar = model->addJoint(root, pin::JointModelPlanar(),
                                  pin::SE3::Identity(), "planar");
    auto continuous = model->addJoint(planar, pin::JointModelRUBZ(),
                                      pin::SE3::Identity(), "continuous");
    model->addFrame(pin::Frame(
        "tool", continuous, 0,
        pin::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.2, -.3, .4)),
        pin::OP_FRAME));
    auto space = std::make_shared<PinocchioConfiguration>(model);
    Vector q = pin::integrate(*model, pin::neutral(*model),
                              Vector::LinSpaced(model->nv, -.1, .3));
    q.segment<4>(3) *= 1.2;
    q.segment<2>(model->joints[planar].idx_q() + 2) *= .8;
    q.segment<2>(model->joints[continuous].idx_q()) *= 1.1;
    PinocchioFramePose pose(model, space, "tool");
    Matrix jac(7, model->nq), numerical(7, model->nq);
    pose.jacobian(q, jac);
    for (int k = 0; k < q.size(); ++k)
    {
        Vector plus = q, minus = q, a(7), b(7);
        plus[k] += 1e-7;
        minus[k] -= 1e-7;
        pose.evaluate(plus, a);
        pose.evaluate(minus, b);
        numerical.col(k) = (a - b) / (2e-7);
    }
    EXPECT_LT((jac - numerical).norm(), 3e-8);
}

TEST(FrameKinematics, ValidationAndCloneOwnership)
{
    Robot robot;
    EXPECT_THROW(PinocchioFramePositions(robot.model, robot.space, {"missing"}),
                 std::invalid_argument);
    EXPECT_THROW(PinocchioFramePositions(robot.model, robot.space, {}),
                 std::invalid_argument);
    EXPECT_THROW(PinocchioFramePose(nullptr, robot.space, "tool"),
                 std::invalid_argument);
    auto workspace = std::make_shared<PinocchioModelWorkspace>(robot.model);
    PinocchioFramePoseError source(robot.model, robot.space, "tool", workspace);
    auto clone = source.clone();
    PinocchioFramePose pose(robot.model, robot.space, "tool");
    Vector target(7);
    pose.evaluate(robot.q, target);
    Vector z(robot.model->nq + 7), result(6);
    z << robot.q, target;
    clone->evaluate(z, result);
    EXPECT_LT(result.norm(), 1e-13);
    Vector invalid = z;
    invalid.segment<4>(3).setZero();
    EXPECT_THROW(clone->evaluate(invalid, result), std::invalid_argument);
    clone->evaluate(z, result);
    EXPECT_LT(result.norm(), 1e-13);
    z.tail<4>().setZero();
    EXPECT_THROW(clone->evaluate(z, result), std::invalid_argument);
    z.tail<4>().setConstant(std::numeric_limits<double>::quiet_NaN());
    EXPECT_THROW(clone->evaluate(z, result), std::invalid_argument);
}

TEST(FrameKinematics, RobotFunctionsShareContextAcrossMixedCalls)
{
    Robot robot;
    auto workspace = std::make_shared<PinocchioModelWorkspace>(robot.model);
    auto context = std::make_shared<EvaluationContext>();
    PinocchioRobotFunction tool(robot.model, robot.space,
                                RobotQuantity::FrameVelocity, "tool",
                                pin::LOCAL_WORLD_ALIGNED, false, workspace);
    PinocchioRobotFunction contact(robot.model, robot.space,
                                   RobotQuantity::FrameVelocity, "contact",
                                   pin::LOCAL_WORLD_ALIGNED, false, workspace);
    PinocchioRobotFunction acceleration(robot.model, robot.space,
                                        RobotQuantity::FrameAcceleration, "tool",
                                        pin::LOCAL_WORLD_ALIGNED, false, workspace);
    PinocchioRobotFunction gravity(robot.model, robot.space, RobotQuantity::Gravity,
                                   {}, pin::LOCAL_WORLD_ALIGNED, false, workspace);
    PinocchioFramePositions positions(robot.model, robot.space, {"tool"}, workspace);
    tool.setEvaluationContext(context);
    contact.setEvaluationContext(context);
    acceleration.setEvaluationContext(context);
    gravity.setEvaluationContext(context);
    positions.setEvaluationContext(context);
    auto independent = tool.clone();
    independent->setEvaluationContext(std::make_shared<EvaluationContext>());
    auto *reference = dynamic_cast<TangentFunction *>(independent.get());
    ASSERT_NE(reference, nullptr);
    Vector z(robot.model->nq + robot.model->nv);
    Vector za(robot.model->nq + 2 * robot.model->nv);
    Vector value(6), expected(6), other(6), position(3), torque(robot.model->nv);
    Matrix jac(6, 2 * robot.model->nv), expected_jac(6, 2 * robot.model->nv);
    Matrix other_jac(6, 2 * robot.model->nv), accel_jac(6, 3 * robot.model->nv);
    for (int trial = 0; trial < 3; ++trial)
    {
        z << pin::integrate(*robot.model, robot.q,
                            Vector::Constant(robot.model->nv, .03 * trial)),
            Vector::LinSpaced(robot.model->nv, -.2 + .1 * trial, .3);
        za << z, Vector::Constant(robot.model->nv, .05 * trial);
        tool.valueAndTangentLinearization(z, value, jac);
        contact.valueAndTangentLinearization(z, other, other_jac);
        positions.evaluate(z.head(robot.model->nq), position);
        acceleration.valueAndTangentLinearization(za, other, accel_jac);
        gravity.evaluate(z.head(robot.model->nq), torque);
        tool.valueAndTangentLinearization(z, value, jac);
        reference->valueAndTangentLinearization(z, expected, expected_jac);
        EXPECT_LT((value - expected).norm(), 1e-12);
        EXPECT_LT((jac - expected_jac).norm(), 1e-12);
        EXPECT_LT((jac - numericalTangent(tool, z)).norm(), 3e-8);
    }
    auto other_context = std::make_shared<EvaluationContext>();
    EXPECT_NE(workspace->forContext(context).get(),
              workspace->forContext(other_context).get());
}
