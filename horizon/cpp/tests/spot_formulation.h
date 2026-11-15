#pragma once
namespace spot_test
{
constexpr int nq=19, nv=18, nf=12, nx=nq+nv;
enum class Formulation { AccelerationForce, Torque };
inline int controlSize(Formulation f) { return f==Formulation::Torque ? 12 : nv+nf; }
inline const char* name(Formulation f) { return f==Formulation::Torque ? "torque" : "acceleration_force"; }
}
