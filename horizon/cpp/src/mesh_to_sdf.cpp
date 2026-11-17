#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/MeshToVolume.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace
{
using Points = std::vector<openvdb::Vec3s>;
using Triangles = std::vector<openvdb::Vec3I>;

void loadStl(const std::string& path, Points& points, Triangles& triangles)
{
    std::ifstream in(path, std::ios::binary);
    if(!in) throw std::runtime_error("cannot open STL: " + path);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
    auto append = [&](const openvdb::Vec3s& a, const openvdb::Vec3s& b, const openvdb::Vec3s& c)
    {
        const int n = static_cast<int>(points.size());
        points.insert(points.end(), {a, b, c});
        triangles.emplace_back(n, n + 1, n + 2);
    };
    std::uint32_t count = 0;
    if(bytes.size() >= 84) std::memcpy(&count, bytes.data() + 80, sizeof(count));
    if(bytes.size() == 84ull + 50ull * count)
    {
        for(std::uint32_t i = 0; i < count; ++i)
        {
            const char* p = bytes.data() + 84 + 50ull * i + 12;
            openvdb::Vec3s v[3];
            for(int k = 0; k < 3; ++k)
            {
                float xyz[3]; std::memcpy(xyz, p + k * 12, 12);
                v[k] = openvdb::Vec3s(xyz[0], xyz[1], xyz[2]);
            }
            append(v[0], v[1], v[2]);
        }
    }
    else
    {
        std::istringstream text(std::string(bytes.begin(), bytes.end()));
        std::string token; openvdb::Vec3s v[3]; int n = 0;
        while(text >> token)
            if(token == "vertex")
            {
                text >> v[n][0] >> v[n][1] >> v[n][2];
                if(++n == 3) { append(v[0], v[1], v[2]); n = 0; }
            }
        if(points.empty()) throw std::runtime_error("STL contains no triangles: " + path);
    }
}
}

int main(int argc, char** argv)
{
    if(argc != 5)
    {
        std::cerr << "usage: horizon_mesh_to_sdf input.stl output.vdb voxel_size band_voxels\n";
        return 2;
    }
    try
    {
        const double voxel = std::stod(argv[3]), band = std::stod(argv[4]);
        if(!std::isfinite(voxel) || voxel <= 0. || !std::isfinite(band) || band < 3.)
            throw std::invalid_argument("voxel size must be positive and band width at least 3 voxels");
        openvdb::initialize();
        Points points; Triangles triangles; loadStl(argv[1], points, triangles);
        auto transform = openvdb::math::Transform::createLinearTransform(voxel);
        auto grid = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
            *transform, points, triangles, std::vector<openvdb::Vec4I>{}, band);
        grid->setName("sdf"); grid->setGridClass(openvdb::GRID_LEVEL_SET);
        openvdb::io::File file(argv[2]); file.write({grid}); file.close();
        std::cout << "wrote " << argv[2] << " from " << triangles.size() << " triangles\n";
    }
    catch(const std::exception& e)
    {
        std::cerr << "mesh to SDF failed: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
