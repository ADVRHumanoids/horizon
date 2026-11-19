#pragma once
#include "example_paths.h"
#include "horizon/pinocchio_kinematics.h"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>

namespace examples
{
inline horizon::ilqr::Vector standing(const pinocchio::Model &model,
                                      bool collision_free = false)
{
    horizon::ilqr::Vector q = pinocchio::neutral(model);
    auto set = [&](const std::string &name, double value)
    {
        q[model.joints[model.getJointId(name)].idx_q()] = value;
    };
    for (int leg = 1; leg <= 4; ++leg)
    {
        double sign = leg % 2 ? 1 : -1;
        set("hip_pitch_" + std::to_string(leg), -.7 * sign);
        set("knee_pitch_" + std::to_string(leg), 1.4 * sign);
        set("hip_roll_" + std::to_string(leg),
            leg == 1 || leg == 4 ? .15 : -.15);
    }
    for (int arm = 1; arm <= 2; ++arm)
    {
        double sign = arm == 1 ? 1 : -1;
        set("shoulder_pitch_" + std::to_string(arm), 1.3 * sign);
        set("elbow_pitch_" + std::to_string(arm), 2.5 * sign);
        set("wrist_pitch_" + std::to_string(arm), .45 * sign);
    }
    if (collision_free)
    {
        set("shoulder_yaw_1", .8);
        set("shoulder_pitch_1", 1.);
        set("elbow_pitch_1", 2.);
        set("wrist_pitch_1", 1.2);
        set("wrist_yaw_1", .5);
        set("shoulder_yaw_2", 1.);
        set("shoulder_pitch_2", -1.15);
        set("elbow_pitch_2", -1.35);
        set("wrist_pitch_2", -.65);
        set("wrist_yaw_2", -.75);
    }
    return q;
}
} // namespace examples
