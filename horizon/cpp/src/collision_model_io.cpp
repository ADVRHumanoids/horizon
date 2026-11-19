#include "horizon/collision.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tinyxml2.h>
#include <yaml-cpp/yaml.h>

namespace horizon::collision
{
SphereModel loadSphereModelFromUrdf(const std::string &path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
    {
        throw std::invalid_argument("cannot parse sphere URDF '" + path +
                                    "': " + doc.ErrorStr());
    }
    const auto *robot = doc.FirstChildElement("robot");
    if (!robot)
    {
        throw std::invalid_argument("URDF has no robot element: " + path);
    }
    std::vector<LinkSphere> spheres;
    std::set<std::string> links;
    for (auto *link = robot->FirstChildElement("link"); link;
         link = link->NextSiblingElement("link"))
    {
        const char *name = link->Attribute("name");
        if (!name || !*name || !links.insert(name).second)
        {
            throw std::invalid_argument(
                "URDF has a missing or duplicate link name");
        }
        for (auto *collision = link->FirstChildElement("collision"); collision;
             collision = collision->NextSiblingElement("collision"))
        {
            const auto *geometry = collision->FirstChildElement("geometry");
            const auto *sphere =
                geometry ? geometry->FirstChildElement("sphere") : nullptr;
            if (!sphere)
            {
                throw std::invalid_argument(
                    "non-sphere collision geometry on link " +
                    std::string(name));
            }
            LinkSphere item;
            item.link = name;
            if (sphere->QueryDoubleAttribute("radius", &item.radius) !=
                    tinyxml2::XML_SUCCESS ||
                !std::isfinite(item.radius) || item.radius <= 0)
            {
                throw std::invalid_argument("invalid sphere radius on link " +
                                            item.link);
            }
            if (auto *origin = collision->FirstChildElement("origin"))
            {
                if (const char *xyz = origin->Attribute("xyz"))
                {
                    std::istringstream input(xyz);
                    std::string extra;
                    if (!(input >> item.center.x() >> item.center.y() >>
                          item.center.z()) ||
                        (input >> extra) || !item.center.allFinite())
                    {
                        throw std::invalid_argument(
                            "invalid sphere origin on link " + item.link);
                    }
                }
            }
            spheres.push_back(std::move(item));
        }
    }
    if (spheres.empty())
    {
        throw std::invalid_argument("URDF contains no collision spheres: " +
                                    path);
    }
    return SphereModel(std::move(spheres));
}

std::vector<DisabledCollisionPair>
loadDisabledCollisionPairs(const std::string &path)
{
    const YAML::Node document = YAML::LoadFile(path);
    if (!document.IsMap() || !document["format"] ||
        document["format"].as<std::string>() !=
            "horizon.disabled_collision_pairs.v1" ||
        !document["disabled_pairs"] || !document["disabled_pairs"].IsSequence())
    {
        throw std::invalid_argument("invalid disabled collision policy: " +
                                    path);
    }
    std::vector<DisabledCollisionPair> result;
    std::set<std::pair<std::string, std::string>> seen;
    for (const auto &node : document["disabled_pairs"])
    {
        if (!node.IsMap() || !node["link1"] || !node["link2"] ||
            !node["reason"])
        {
            throw std::invalid_argument("invalid disabled collision pair: " +
                                        path);
        }
        DisabledCollisionPair pair{node["link1"].as<std::string>(),
                                   node["link2"].as<std::string>(),
                                   node["reason"].as<std::string>()};
        if (pair.link1.empty() || pair.link2.empty() || pair.reason.empty() ||
            pair.link1 == pair.link2)
        {
            throw std::invalid_argument("invalid disabled collision pair: " +
                                        path);
        }
        auto key = std::minmax(pair.link1, pair.link2);
        if (!seen.emplace(key.first, key.second).second)
        {
            throw std::invalid_argument("duplicate disabled collision pair: " +
                                        path);
        }
        result.push_back(std::move(pair));
    }
    return result;
}

std::vector<std::pair<int, int>>
makeSelfCollisionPairs(const SphereModel &model,
                       const std::vector<DisabledCollisionPair> &disabled)
{
    std::set<std::pair<std::string, std::string>> excluded;
    std::set<std::string> links;
    for (const auto &sphere : model.spheres())
    {
        links.insert(sphere.link);
    }
    for (const auto &pair : disabled)
    {
        if (!links.count(pair.link1) || !links.count(pair.link2))
        {
            throw std::invalid_argument(
                "disabled collision pair references a link without spheres: " +
                pair.link1 + ", " + pair.link2);
        }
        auto key = std::minmax(pair.link1, pair.link2);
        excluded.emplace(key.first, key.second);
    }
    const auto &spheres = model.spheres();
    std::vector<std::pair<int, int>> result;
    for (std::size_t i = 0; i < spheres.size(); ++i)
    {
        for (std::size_t j = i + 1; j < spheres.size(); ++j)
        {
            if (spheres[i].link == spheres[j].link)
            {
                continue;
            }
            auto key = std::minmax(spheres[i].link, spheres[j].link);
            if (!excluded.count({key.first, key.second}))
            {
                result.emplace_back(i, j);
            }
        }
    }
    return result;
}
} // namespace horizon::collision
