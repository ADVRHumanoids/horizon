#include "horizon/collision.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <set>

using namespace horizon::collision;

TEST(CollisionModel, UrdfAndPolicyCoverEveryEnabledSpherePair)
{
    const auto model =
        loadSphereModelFromUrdf("horizon/cpp/tests/data/kyon_spheres.urdf");
    const auto disabled = loadDisabledCollisionPairs(
        "horizon/cpp/tests/data/kyon_disabled_collisions.json");
    const auto pairs = makeSelfCollisionPairs(model, disabled);
    ASSERT_EQ(model.spheres().size(), 96);
    ASSERT_GT(disabled.size(), 0);
    ASSERT_GT(pairs.size(), 0);
    std::set<std::pair<std::string, std::string>> exclusions;
    for (const auto &pair : disabled)
    {
        auto links = std::minmax(pair.link1, pair.link2);
        exclusions.emplace(links.first, links.second);
    }
    std::set<std::pair<int, int>> actual(pairs.begin(), pairs.end());
    ASSERT_EQ(actual.size(), pairs.size());
    const auto &spheres = model.spheres();
    for (std::size_t i = 0; i < spheres.size(); ++i)
    {
        for (std::size_t j = i + 1; j < spheres.size(); ++j)
        {
            const auto links = std::minmax(spheres[i].link, spheres[j].link);
            const bool expected = spheres[i].link != spheres[j].link &&
                !exclusions.count({links.first, links.second});
            EXPECT_EQ(actual.count({int(i), int(j)}) != 0, expected)
                << i << "," << j;
        }
    }
}

TEST(CollisionModel, FullBodyFixtureHasEveryXacroCollisionLink)
{
    const auto model = loadSphereModelFromUrdf(
        "horizon/cpp/tests/data/kyon_full_spheres.urdf");
    const auto disabled = loadDisabledCollisionPairs(
        "horizon/cpp/tests/data/kyon_full_disabled_collisions.json");
    const auto pairs = makeSelfCollisionPairs(model, disabled);
    std::set<std::string> links;
    for (const auto &sphere : model.spheres())
    {
        links.insert(sphere.link);
    }
    EXPECT_EQ(model.spheres().size(), 103);
    EXPECT_EQ(links.size(), 31);
    EXPECT_TRUE(links.count("shoulder_pitch_1_link"));
    EXPECT_TRUE(links.count("elbow_pitch_2_link"));
    EXPECT_EQ(pairs.size(), 3799);
}

TEST(CollisionModel, RejectsMissingInputs)
{
    EXPECT_THROW(loadSphereModelFromUrdf("/no/such/robot.urdf"),
                 std::exception);
    EXPECT_THROW(loadDisabledCollisionPairs("/no/such/collision-policy.json"),
                 std::exception);
}

TEST(CollisionModel, RejectsMismatchedOrDuplicatePolicies)
{
    SphereModel model({{"a", Eigen::Vector3d::Zero(), .1},
                       {"b", Eigen::Vector3d::Ones(), .2}});
    EXPECT_THROW(makeSelfCollisionPairs(model, {{"a", "unknown", "Never"}}),
                 std::invalid_argument);
    const auto path = std::filesystem::temp_directory_path() /
        "horizon_duplicate_collision_policy.json";
    {
        std::ofstream out(path);
        out << R"({"format":"horizon.disabled_collision_pairs.v1","disabled_pairs":[)"
            << R"({"link1":"a","link2":"b","reason":"Never"},)"
            << R"({"link1":"b","link2":"a","reason":"Adjacent"}]})";
    }
    EXPECT_THROW(loadDisabledCollisionPairs(path.string()),
                 std::invalid_argument);
    std::filesystem::remove(path);
}
