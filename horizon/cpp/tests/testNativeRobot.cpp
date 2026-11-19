#include "horizon/pinocchio_robot.h"
#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>

using namespace horizon::ilqr;
using namespace horizon::kinematics;
namespace pin = pinocchio;
namespace
{
struct Robot
{
    std::shared_ptr<pin::Model> model = std::make_shared<pin::Model>();
    std::shared_ptr<PinocchioConfiguration> space;
    Vector q, v, a, tau;
    pin::FrameIndex tool, foot;
    Robot()
    {
        auto root = model->addJoint(0, pin::JointModelFreeFlyer(),
                                    pin::SE3::Identity(), "base");
        model->appendBodyToJoint(root,
                                 pin::Inertia(5., Eigen::Vector3d(.1, 0, .2),
                                              Eigen::Matrix3d::Identity()),
                                 pin::SE3::Identity());
        auto arm = model->addJoint(root, pin::JointModelRY(),
                                   pin::SE3::Identity(), "arm");
        model->appendBodyToJoint(arm,
                                 pin::Inertia(2., Eigen::Vector3d(.4, .1, .2),
                                              .3 * Eigen::Matrix3d::Identity()),
                                 pin::SE3::Identity());
        tool = model->addFrame(pin::Frame(
            "tool", arm, 0,
            pin::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(.6, 0, .1)),
            pin::OP_FRAME));
        foot =
            model->addFrame(pin::Frame("foot", root, 0,
                                       pin::SE3(Eigen::Matrix3d::Identity(),
                                                Eigen::Vector3d(.2, -.3, -.2)),
                                       pin::OP_FRAME));
        space = std::make_shared<PinocchioConfiguration>(model);
        q = pin::integrate(*model, pin::neutral(*model),
                           Vector::LinSpaced(model->nv, -.2, .3));
        v = Vector::LinSpaced(model->nv, -.3, .2);
        a = Vector::LinSpaced(model->nv, .2, -.1);
        tau = Vector::LinSpaced(model->nv, -1., 2.);
    }
};
void checkJacobian(TangentFunction &function, const Vector &input,
                   double tolerance = 2e-6)
{
    const auto &space = function.tangentInputSpace();
    Matrix jac(function.outputSize(), space.tangentSize()),
        numerical(jac.rows(), jac.cols());
    Vector value(function.outputSize());
    function.valueAndTangentLinearization(input, value, jac);
    Vector delta = Vector::Zero(space.tangentSize()), plus(input.size()),
           minus(input.size()), p(value.size()), m(value.size());
    for (int k = 0; k < space.tangentSize(); ++k)
    {
        delta[k] = 1e-6;
        space.integrate(input, delta, plus);
        delta[k] = -1e-6;
        space.integrate(input, delta, minus);
        delta[k] = 0;
        function.evaluate(plus, p);
        function.evaluate(minus, m);
        numerical.col(k) = (p - m) / 2e-6;
    }
    EXPECT_LT((jac - numerical).norm(), tolerance)
        << "shape " << jac.rows() << " x " << jac.cols();
}
Vector inputFor(Robot &robot, RobotQuantity quantity, Function &function)
{
    Vector input(function.inputSize());
    input.head(robot.model->nq) = robot.q;
    int remaining = input.size() - robot.model->nq;
    if (quantity == RobotQuantity::Difference)
    {
        input.tail(robot.model->nq) =
            pin::integrate(*robot.model, robot.q, .1 * robot.v);
        return input;
    }
    if (remaining >= robot.model->nv)
    {
        input.segment(robot.model->nq, robot.model->nv) = robot.v;
    }
    if (remaining >= 2 * robot.model->nv)
    {
        input.segment(robot.model->nq + robot.model->nv, robot.model->nv) =
            quantity == RobotQuantity::ForwardDynamics ? robot.tau : robot.a;
    }
    return input;
}
} // namespace

TEST(NativeRobot, AllAnalyticalQuantityJacobians)
{
    Robot robot;
    for (auto quantity :
         {RobotQuantity::QDot, RobotQuantity::Difference,
          RobotQuantity::FrameTransform, RobotQuantity::InverseDynamics,
          RobotQuantity::ForwardDynamics, RobotQuantity::MassMatrix,
          RobotQuantity::Gravity, RobotQuantity::NonlinearEffects,
          RobotQuantity::KineticEnergy, RobotQuantity::PotentialEnergy,
          RobotQuantity::CenterOfMass, RobotQuantity::ComPosition,
          RobotQuantity::ComVelocity, RobotQuantity::ComAcceleration,
          RobotQuantity::CentroidalMomentum, RobotQuantity::CentroidalDynamics,
          RobotQuantity::CentroidalMap})
    {
        SCOPED_TRACE(static_cast<int>(quantity));
        PinocchioRobotFunction function(robot.model, robot.space, quantity,
                                        "tool");
        checkJacobian(function, inputFor(robot, quantity, function));
    }
    for (auto reference : {pin::LOCAL, pin::WORLD, pin::LOCAL_WORLD_ALIGNED})
    {
        for (auto quantity :
             {RobotQuantity::FrameJacobian, RobotQuantity::FrameVelocity,
              RobotQuantity::FrameAcceleration,
              RobotQuantity::FrameSpatialAcceleration})
        {
            SCOPED_TRACE(std::to_string(static_cast<int>(quantity)) + " ref " +
                         std::to_string(reference));
            PinocchioRobotFunction function(robot.model, robot.space, quantity,
                                            "tool", reference);
            checkJacobian(function, inputFor(robot, quantity, function));
        }
    }
}

TEST(NativeRobot, DynamicsAndComValuesMatchIndependentPinocchio)
{
    Robot robot;
    pin::Data data(*robot.model);
    auto evaluate = [&](RobotQuantity quantity)
    {
        PinocchioRobotFunction f(robot.model, robot.space, quantity);
        Vector out(f.outputSize());
        f.evaluate(inputFor(robot, quantity, f), out);
        return out;
    };
    EXPECT_LT((evaluate(RobotQuantity::InverseDynamics) -
               pin::rnea(*robot.model, data, robot.q, robot.v, robot.a))
                  .norm(),
              1e-12);
    EXPECT_LT((evaluate(RobotQuantity::ForwardDynamics) -
               pin::aba(*robot.model, data, robot.q, robot.v, robot.tau))
                  .norm(),
              1e-12);
    Matrix M = pin::crba(*robot.model, data, robot.q);
    M.triangularView<Eigen::StrictlyLower>() =
        M.transpose().triangularView<Eigen::StrictlyLower>();
    EXPECT_LT((evaluate(RobotQuantity::MassMatrix) -
               Eigen::Map<const Vector>(M.data(), M.size()))
                  .norm(),
              1e-12);
    pin::centerOfMass(*robot.model, data, robot.q, robot.v, robot.a);
    Vector com(9);
    com << data.com[0], data.vcom[0], data.acom[0];
    EXPECT_LT((evaluate(RobotQuantity::CenterOfMass) - com).norm(), 1e-12);
    pin::computeCentroidalMomentumTimeVariation(*robot.model, data, robot.q,
                                                robot.v, robot.a);
    Vector centroidal(12);
    centroidal << data.hg.toVector(), data.dhg.toVector();
    EXPECT_LT((evaluate(RobotQuantity::CentroidalDynamics) - centroidal).norm(),
              1e-12);
    PinocchioRobotFunction symbolic_gravity(robot.model, robot.space,
                                            RobotQuantity::InverseDynamics, {},
                                            pin::LOCAL_WORLD_ALIGNED, true);
    Vector input(symbolic_gravity.inputSize());
    input << robot.q, robot.v, robot.a, Eigen::Vector3d(0, 0, -3.);
    checkJacobian(symbolic_gravity, input);
}

TEST(NativeRobot, ContactForwardInverseAndActuationJacobians)
{
    Robot robot;
    for (int dimension : {3, 6})
    {
        for (bool actuated : {false, true})
        {
            SCOPED_TRACE(std::to_string(dimension) + " actuated " +
                         std::to_string(actuated));
            Matrix S = Matrix::Identity(robot.model->nv, robot.model->nv);
            if (actuated)
            {
                S = S.rightCols(1).eval();
            }
            PinocchioContactDynamics function(robot.model, robot.space,
                                              {{"foot", dimension}}, S);
            Vector input(function.inputSize());
            input << robot.q, robot.v, Vector::LinSpaced(S.cols(), -.2, .3);
            Vector result(function.outputSize());
            function.evaluate(input, result);
            checkJacobian(function, input, 2e-5);
            pin::Data data(*robot.model);
            Matrix J = Matrix::Zero(6, robot.model->nv);
            pin::forwardKinematics(*robot.model, data, robot.q, robot.v,
                                   Vector::Zero(robot.model->nv));
            pin::computeJointJacobians(*robot.model, data);
            pin::updateFramePlacements(*robot.model, data);
            pin::getFrameJacobian(*robot.model, data, robot.foot,
                                  pin::LOCAL_WORLD_ALIGNED, J);
            Vector gamma =
                pin::getFrameClassicalAcceleration(
                    *robot.model, data, robot.foot, pin::LOCAL_WORLD_ALIGNED)
                    .toVector()
                    .head(dimension);
            EXPECT_LT(
                (J.topRows(dimension) * result.head(robot.model->nv) + gamma)
                    .norm(),
                1e-10);
            Vector required = pin::rnea(*robot.model, data, robot.q, robot.v,
                                        result.head(robot.model->nv));
            EXPECT_LT(
                (required - S * input.tail(S.cols()) -
                 J.topRows(dimension).transpose() * result.tail(dimension))
                    .norm(),
                1e-10);
            PinocchioContactDynamics inverse(robot.model, robot.space,
                                             {{"foot", dimension}}, Matrix(),
                                             true);
            Vector inverse_input(inverse.inputSize());
            inverse_input << robot.q, robot.v, result.head(robot.model->nv),
                result.tail(dimension);
            Vector torque(robot.model->nv);
            inverse.evaluate(inverse_input, torque);
            EXPECT_LT((torque - S * input.tail(S.cols())).norm(), 1e-10);
            checkJacobian(inverse, inverse_input, 2e-5);
        }
    }
}

TEST(NativeRobot, EmptyContactsAndInvalidSets)
{
    Robot robot;
    PinocchioContactDynamics free(robot.model, robot.space, {});
    Vector input(free.inputSize());
    input << robot.q, robot.v, robot.tau;
    Vector result(free.outputSize());
    free.evaluate(input, result);
    pin::Data data(*robot.model);
    EXPECT_LT(
        (result - pin::aba(*robot.model, data, robot.q, robot.v, robot.tau))
            .norm(),
        1e-10);
    EXPECT_THROW(
        PinocchioContactDynamics(robot.model, robot.space, {{"missing", 3}}),
        std::invalid_argument);
    EXPECT_THROW(
        PinocchioContactDynamics(robot.model, robot.space, {{"foot", 4}}),
        std::invalid_argument);
    EXPECT_THROW(PinocchioContactDynamics(robot.model, robot.space,
                                          {{"foot", 3}, {"foot", 6}}),
                 std::invalid_argument);
    EXPECT_THROW(PinocchioContactDynamics(nullptr, robot.space, {}),
                 std::invalid_argument);
    PinocchioContactDynamics singular(robot.model, robot.space,
                                      {{"foot", 6}, {"tool", 6}});
    Vector output(singular.outputSize());
    EXPECT_THROW(singular.evaluate(input, output), EvaluationError);
}

TEST(NativeRobot, QdotCompositePlanarContinuousAndSphericalJoints)
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
    model->addJoint(continuous, pin::JointModelSpherical(),
                    pin::SE3::Identity(), "spherical");
    auto space = std::make_shared<PinocchioConfiguration>(model);
    Vector q = pin::integrate(*model, pin::neutral(*model),
                              Vector::LinSpaced(model->nv, -.2, .3));
    Vector v = Vector::LinSpaced(model->nv, .1, -.4),
           input(q.size() + v.size());
    input << q, v;
    PinocchioRobotFunction function(model, space, RobotQuantity::QDot);
    checkJacobian(function, input);
    Vector value(q.size());
    function.evaluate(input, value);
    Vector expected = (pin::integrate(*model, q, (1e-7 * v).eval()) -
                       pin::integrate(*model, q, (-1e-7 * v).eval())) /
        (2e-7);
    EXPECT_LT((value - expected).norm(), 1e-8);
}
