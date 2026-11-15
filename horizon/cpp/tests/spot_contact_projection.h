#pragma once
#include "spot_model.h"

namespace spot_test
{
/** Fixed-foot projection preserving the raw base pose and base velocity.
 * Joint positions come from IK, joint velocities from J(q)*v=0. The derivative
 * follows these two implicit equations using native Pinocchio Jacobians. */
class ContactProjection
{
public:
    explicit ContactProjection(std::shared_ptr<const Robot>);
    ~ContactProjection();
    void evaluate(Input raw,VectorOutput projected);
    // Columns are raw configuration tangent and raw generalized velocity.
    void valueAndJacobian(Input raw,VectorOutput projected,MatrixOutput jac);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
