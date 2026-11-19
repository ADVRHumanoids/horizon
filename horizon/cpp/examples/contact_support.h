#pragma once
#include "horizon/pinocchio_robot.h"
#include "support.h"
#include <Eigen/QR>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace examples
{
// Find one static combination of joint torques and point-contact forces.
inline horizon::ilqr::Vector
equilibrium(const pinocchio::Model &model, const horizon::ilqr::Vector &q,
            const horizon::ilqr::Matrix &selection,
            const std::vector<horizon::kinematics::ContactSpec> &contacts)
{
    using horizon::ilqr::Matrix;
    pinocchio::Data data(model);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    Matrix jacobian(3 * contacts.size(), model.nv);
    for (std::size_t i = 0; i < contacts.size(); ++i)
    {
        Matrix frame = Matrix::Zero(6, model.nv);
        pinocchio::getFrameJacobian(model, data,
                                    model.getFrameId(contacts[i].frame),
                                    pinocchio::LOCAL_WORLD_ALIGNED, frame);
        jacobian.middleRows(3 * i, 3) = frame.topRows(3);
    }
    Matrix balance(model.nv, selection.cols() + jacobian.rows());
    balance << selection, jacobian.transpose();
    return balance.completeOrthogonalDecomposition().solve(
        pinocchio::computeGeneralizedGravity(model, data, q));
}
} // namespace examples
