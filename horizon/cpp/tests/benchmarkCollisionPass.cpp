// Controlled reference for tools/probe_curobo_v2.py (same q and SDF samples).
#include "horizon/device_collision.h"
#include "horizon/pinocchio_collision.h"
#include "horizon/kyon_spheres.h"
#include <Kokkos_Core.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <filesystem>

using namespace horizon::collision;
template<class T> std::vector<T> read(const std::filesystem::path& p,std::size_t count)
{
    std::ifstream input(p,std::ios::binary);std::vector<T> values(count);
    if(!input.read(reinterpret_cast<char*>(values.data()),count*sizeof(T)) || input.peek()!=EOF)
        throw std::runtime_error("invalid reference data: "+p.string());
    return values;
}
double median(std::vector<double> v){std::sort(v.begin(),v.end());return .5*(v[(v.size()-1)/2]+v[v.size()/2]);}
int main(int argc,char** argv)
{
    try
    {
        if(argc!=2)throw std::invalid_argument("usage: benchmarkCollisionPass REFERENCE_DIRECTORY");
        Kokkos::ScopeGuard runtime;
        const std::filesystem::path directory(argv[1]);const auto input=YAML::LoadFile((directory/"manifest.json").string());
        pinocchio::Model model;pinocchio::urdf::buildModel(input["urdf"].as<std::string>(),model);
        Eigen::Vector3i dims;Eigen::Vector3d origin;for(int a=0;a<3;++a){dims[a]=input["dimensions"][a].as<int>();origin[a]=input["origin"][a].as<double>();}
        auto grid=std::make_shared<SdfGrid>(dims,origin,input["voxel_size"].as<double>(),read<float>(directory/"grid.f32",dims.prod()));
        const auto geometry=kyon::spheres();std::vector<SphereDistanceFunction::Pair> pairs;
        for(int i=0;i<int(geometry.size());++i)for(int j=i+1;j<int(geometry.size());++j)
            if(geometry[i].link.find("dagana_1")==0 && geometry[j].link.find("dagana_2")==0)pairs.emplace_back(i,j);
        DeviceCollisionCost cost(exportDeviceRobot(model,SphereModel(geometry)),grid,pairs,10000.);
        std::cout<<std::setprecision(12)<<"{\"execution_space\":\""<<DeviceCollisionCost::executionSpace()<<"\",\"cases\":[";
        bool first=true;const int entries=1+model.nv+model.nv*(model.nv+1)/2;
        for(auto item:input["batches"])
        {
            const int n=item.as<int>();const auto values=read<double>(directory/("q"+std::to_string(n)+".f64"),n*model.nq);
            const auto expected=read<double>(directory/("expected"+std::to_string(n)+".f64"),n*entries);
            Eigen::MatrixXd states(model.nq,n);for(int k=0;k<n;++k)for(int a=0;a<model.nq;++a)states(a,k)=values[k*model.nq+a];
            cost.prepare(states,true);double error=0;
            for(int k=0;k<n;++k)
            {
                auto compare=[&](double actual,int entry){const double ref=expected[k*entries+entry];error=std::max(error,std::abs(actual-ref)/(1+std::abs(ref)));};
                compare(cost.value(k),0);for(int a=0;a<model.nv;++a)compare(cost.gradient(k)[a],1+a);
                for(int a=0;a<entries-1-model.nv;++a)compare(cost.packedHessian(k)[a],1+model.nv+a);
            }
            if(error>1e-7)throw std::runtime_error("FP64 reference parity failed");
            for(int k=0;k<5;++k)cost.prepare(states,true);
            std::vector<double> gpu,wall,fk,query,gn;
            for(int k=0;k<20;++k)
            {
                cost.resetStatistics();cost.prepare(states,true);const auto s=cost.statistics();
                gpu.push_back(s.kinematics_ms+s.queries_ms+s.accumulation_ms);wall.push_back(s.prepare_ms);
                fk.push_back(s.kinematics_ms);query.push_back(s.queries_ms);gn.push_back(s.accumulation_ms);
            }
            if(!first)std::cout<<',';first=false;
            std::cout<<"{\"nodes\":"<<n<<",\"oracle_max_scaled_error\":"<<error<<",\"gpu_compute_median_ms\":"<<median(gpu)
                     <<",\"prepare_wall_median_ms\":"<<median(wall)<<",\"fk_median_ms\":"<<median(fk)
                     <<",\"queries_median_ms\":"<<median(query)<<",\"gn_median_ms\":"<<median(gn)<<'}';
        }
        std::cout<<"]}\n";
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
