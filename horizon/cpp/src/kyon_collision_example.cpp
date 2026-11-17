// Kyon fixed-feet iLQR example and reproducible collision-backend benchmark.
#include "horizon/kyon_spheres.h"
#include "horizon/pinocchio_collision.h"
#include "ilqr/ilqr.h"
#include "ilqr/casadi_function.h"
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <sstream>
#ifndef HORIZON_KYON_URDF_PATH
#define HORIZON_KYON_URDF_PATH ""
#endif
#ifdef HORIZON_WITH_KOKKOS
#include "horizon/kokkos_collision.h"
#include <Kokkos_Core.hpp>
#endif

using namespace horizon::ilqr;
using namespace horizon::collision;
namespace pin = pinocchio;
using Clock = std::chrono::steady_clock;

namespace
{
using Configuration = PinocchioConfiguration;
using Step = PinocchioStep;

struct Counters { std::atomic<unsigned long long> values{0}, products{0}, nanos{0}; };
// Time complete collision calls, never individual sphere queries.
class TimedCollision final : public Function
{
public:
    TimedCollision(std::shared_ptr<const Function> f,std::shared_ptr<Counters> c):
        Function(f->inputSize(),f->outputSize()),prototype(f),inner(f->clone()),counts(c) {}
    std::unique_ptr<Function> clone() const override { return std::make_unique<TimedCollision>(prototype,counts); }
    bool hasJacobianProduct() const override { return prototype->hasJacobianProduct(); }
private:
    void evaluateImpl(Input x,VectorOutput y) override
    { const auto t=Clock::now(); inner->evaluate(x,y); record(t,false); }
    void jacobianImpl(Input x,MatrixOutput j) override
    { const auto t=Clock::now(); inner->jacobian(x,j); record(t,true); }
    void valueAndJacobianImpl(Input x,VectorOutput y,MatrixOutput j) override
    { const auto t=Clock::now(); inner->valueAndJacobian(x,y,j); record(t,true); }
    void valueAndJacobianProductImpl(Input x,MatrixInput d,VectorOutput y,MatrixOutput p) override
    { const auto t=Clock::now(); inner->valueAndJacobianProduct(x,d,y,p); record(t,true); }
    void record(Clock::time_point t,bool derivative)
    {
        counts->nanos.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count(),std::memory_order_relaxed);
        (derivative?counts->products:counts->values).fetch_add(1,std::memory_order_relaxed);
    }
    std::shared_ptr<const Function> prototype;
    std::unique_ptr<Function> inner;
    std::shared_ptr<Counters> counts;
};

casadi::MX vectorConstant(const Vector& v)
{ return casadi::MX(casadi::DM(std::vector<double>(v.data(),v.data()+v.size()))); }
std::string jsonString(const std::string& value)
{
    std::ostringstream out;out<<'"';
    for(unsigned char c:value)
    {
        if(c=='"' || c=='\\')out<<'\\'<<c;
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;
        else out<<c;
    }
    out<<'"';return out.str();
}
}

int main(int argc,char** argv)
{
    try
    {
        // default values, can be overridden by command-line arguments
        int N = 30, iterations = 40, samples = 3, threads = 0, warmup = 1;
        double damping = .001;
        double collision_weight = -1.;
        std::string decomposition="lu";
        std::string backend="scalar";
        int kokkos_threads=1;
        bool collision_enabled=true;
        bool self_collision=false;
        std::string urdf=HORIZON_KYON_URDF_PATH,scenario="benchmark";
        std::string output="/tmp/kyon_trajectory";
        for(int i=1;i<argc;++i)
        {
            const std::string key=argv[i];
            if(key=="--help")
            {
                std::cout<<"Kyon fixed-feet iLQR: --urdf PATH --nodes N --scenario benchmark|avoidance\n"
                         <<"  --collision 0|1 --self-collision 0|1 --backend scalar|kokkos|device\n"
                         <<"  --samples K --warmup K --iterations K --threads K --kokkos-threads K\n"
                         <<"  --damping VALUE --collision-weight VALUE --decomposition lu|ldlt --output PREFIX\n";return 0;
            }
            if(i+1>=argc) throw std::invalid_argument("missing option value");
            const std::string value=argv[++i];
            if(key=="--nodes") N=std::stoi(value);
            else if(key=="--iterations") iterations=std::stoi(value);
            else if(key=="--samples") samples=std::stoi(value);
            else if(key=="--warmup") warmup=std::stoi(value);
            else if(key=="--urdf") urdf=value;
            else if(key=="--scenario") scenario=value;
            else if(key=="--threads") threads=std::stoi(value);
            else if(key=="--damping") damping=std::stod(value);
            else if(key=="--collision-weight") collision_weight=std::stod(value);
            else if(key=="--decomposition") decomposition=value;
            else if(key=="--backend") backend=value;
            else if(key=="--kokkos-threads") kokkos_threads=std::stoi(value);
            else if(key=="--collision") collision_enabled=std::stoi(value)!=0;
            else if(key=="--self-collision") self_collision=std::stoi(value)!=0;
            else if(key=="--output") output=value;
            else throw std::invalid_argument("unknown option: "+key);
        }
        if(N<4 || iterations<1 || samples<1 || threads<0 || warmup<0 || !std::isfinite(damping) || damping<0.)
            throw std::invalid_argument("invalid benchmark dimensions or damping");
        if(urdf.empty() || !std::filesystem::is_regular_file(urdf))throw std::invalid_argument("provide --urdf /path/to/kyon.urdf");
        if(scenario!="benchmark" && scenario!="avoidance")throw std::invalid_argument("scenario must be benchmark or avoidance");
        if(collision_weight==-1.)collision_weight=scenario=="avoidance"?1000.:10000.;
        if(!std::isfinite(collision_weight) || collision_weight<=0)throw std::invalid_argument("collision weight must be finite and positive");
        const double collision_scale=std::sqrt(collision_weight);
        if((backend!="scalar" && backend!="kokkos" && backend!="device") || kokkos_threads<1)
            throw std::invalid_argument("backend must be scalar, kokkos, or device; Kokkos thread count must be positive");
#ifdef HORIZON_WITH_KOKKOS
        std::unique_ptr<Kokkos::ScopeGuard> kokkos_runtime;
        if(backend!="scalar")kokkos_runtime=std::make_unique<Kokkos::ScopeGuard>(
            Kokkos::InitializationSettings().set_num_threads(kokkos_threads));
#else
        if(backend!="scalar")throw std::invalid_argument("rebuild with HORIZON_WITH_KOKKOS=ON");
#endif
        const auto setup_start=Clock::now();
        auto model=std::make_shared<pin::Model>();
        pin::urdf::buildModel(urdf,*model);
        auto space=std::make_shared<Configuration>(model);
        Vector start=pin::neutral(*model);
        auto setJoint=[&](const std::string& name,double value)
        {
            if(!model->existJointName(name)) throw std::runtime_error("missing joint: "+name);
            const auto& j=model->joints[model->getJointId(name)];
            if(j.nq()!=1) throw std::runtime_error("expected scalar joint: "+name);
            start[j.idx_q()]=value;
        };
        for(int leg=1;leg<=4;++leg)
        {
            const double sign=leg%2 ? 1.:-1.;
            setJoint("hip_pitch_"+std::to_string(leg),-.7*sign);
            setJoint("knee_pitch_"+std::to_string(leg),1.4*sign);
            setJoint("hip_roll_"+std::to_string(leg),(leg==1||leg==4)?.15:-.15);
        }
        for(int arm=1;arm<=2;++arm)
        {
            const double sign=arm==1?1.:-1.;
            setJoint("shoulder_pitch_"+std::to_string(arm),1.3*sign);
            setJoint("elbow_pitch_"+std::to_string(arm),2.5*sign);
            setJoint("wrist_pitch_"+std::to_string(arm),.45*sign);
        }
        // Append frame origins to the same kinematic batch as collision spheres.
        auto geometry=kyon::spheres();
        const int sphere_count=geometry.size();
        for(int f=1;f<=4;++f) geometry.push_back({"contact_"+std::to_string(f),Eigen::Vector3d::Zero(),.001});
        geometry.push_back({"dagana_1_base",Eigen::Vector3d::Zero(),.001});
        const int count=geometry.size();
        auto fk=std::make_shared<PinocchioSphereCenters>(model,space,SphereModel(geometry));
        Vector initial_points(3*count); fk->evaluate(start,initial_points);
        auto point=[&](const Vector& p,int i) { return Eigen::Vector3d(p[i],p[count+i],p[2*count+i]); };
        const Eigen::Vector3d initial_tool=point(initial_points,count-1);
        const Eigen::Vector3d target_offset=scenario=="avoidance"?Eigen::Vector3d(.18,.04,.02):Eigen::Vector3d(.06,.04,.02);
        const Eigen::Vector3d target=initial_tool+target_offset;

        // A sampled sphere SDF gives reproducible trilinear-query work without
        // requiring OpenVDB or mesh conversion in benchmark setup.
        Eigen::Vector3d obstacle=initial_tool+Eigen::Vector3d(.20,0,0);
        const double obstacle_radius=scenario=="avoidance"?.025:.06;
        constexpr double voxel=.04;
        if(scenario=="avoidance")obstacle=point(initial_points,6)+.5*target_offset+Eigen::Vector3d(0,0,.04);
        auto minClearance=[&](const Vector& p,const Eigen::Vector3d& o)
        {
            double result=1e9;
            for(int i=0;i<sphere_count;++i) result=std::min(result,(point(p,i)-o).norm()-obstacle_radius-geometry[i].radius);
            return result;
        };
        for(int attempt=0;minClearance(initial_points,obstacle)<(scenario=="avoidance"?.015:.035);++attempt)
        {
            if(attempt==100) throw std::runtime_error("cannot place obstacle");
            if(scenario=="avoidance")obstacle.z()+=.005;else obstacle.x()+=.02;
        }
        const Eigen::Vector3d origin(-3,-3,-3);
        const Eigen::Vector3i dims(151,151,151);
        std::vector<float> grid(static_cast<std::size_t>(dims.prod()));
        for(int z=0;z<dims.z();++z) for(int y=0;y<dims.y();++y) for(int x=0;x<dims.x();++x)
            grid[(static_cast<std::size_t>(z)*dims.y()+y)*dims.x()+x]=
                (origin+voxel*Eigen::Vector3d(x,y,z)-obstacle).norm()-obstacle_radius;
        auto field=std::make_shared<SdfGrid>(dims,origin,voxel,std::move(grid));
        std::vector<double> radii;
        for(int i=0;i<sphere_count;++i) radii.push_back(geometry[i].radius);
        // Explicit benchmark mask: every sphere on dagana_1 against dagana_2.
        // This is a workload selection, not a complete robot collision mask.
        std::vector<SphereDistanceFunction::Pair> self_pairs;
        if(self_collision)for(int i=0;i<sphere_count;++i)for(int j=i+1;j<sphere_count;++j)
            if(geometry[i].link.find("dagana_1")==0 && geometry[j].link.find("dagana_2")==0)
                self_pairs.emplace_back(i,j);
        auto counters=std::make_shared<Counters>();
        auto collision=std::make_shared<TimedCollision>(std::make_shared<SphereDistanceFunction>(
            radii,std::vector<std::shared_ptr<const DistanceField>>{field},self_pairs),counters);
#ifdef HORIZON_WITH_KOKKOS
        std::unique_ptr<KokkosCollisionBatch> batch;
        std::unique_ptr<DeviceCollisionCost> device;
        if(collision_enabled && backend=="kokkos")batch=std::make_unique<KokkosCollisionBatch>(fk,radii,field,self_pairs);
        if(collision_enabled && backend=="device")device=std::make_unique<DeviceCollisionCost>(
            exportDeviceRobot(*model,SphereModel(std::vector<LinkSphere>(geometry.begin(),geometry.begin()+sphere_count))),field,self_pairs,collision_weight);
#endif
        SymbolicFunctionManager manager;
        using casadi::MX; using casadi::Slice;
        auto q=MX::sym("q",model->nq),u=MX::sym("u",model->nv);
        // The device backend owns collision kinematics. CPU task functions
        // need only five frame origins, not the 94 collision sphere centers.
        std::shared_ptr<const Function> task_fk=fk;
        const bool host_tasks_only=!collision_enabled || backend=="device";
        const int task_count=host_tasks_only?5:count;
        if(host_tasks_only)task_fk=std::make_shared<PinocchioSphereCenters>(model,space,
            SphereModel(std::vector<LinkSphere>(geometry.begin()+sphere_count,geometry.end())));
        auto native_fk=manager.addNative("robot_points",task_fk);
        auto points=native_fk(q);
        auto at=[&](int i) {
            if(host_tasks_only)i-=sphere_count;
            return MX::vertcat({points(i),points(task_count+i),points(2*task_count+i)});
        };
        auto tool=at(count-1);
        std::vector<MX> feet;
        for(int i=0;i<4;++i) feet.push_back(at(sphere_count+i)-vectorConstant(point(initial_points,sphere_count+i)));
        manager.addExpression("feet",{q,u},MX::vertcat(feet),false);
        auto native_step=manager.addNative("integrate",std::make_shared<Step>(space,2./N));
        manager.addExpression("dynamics",{q,u},native_step(MX::vertcat({q,u})),false);
        manager.addResidual("effort",{q,u},.05*u);
        manager.addResidual("target",{q,u},100*(tool-vectorConstant(target)));
        if(collision_enabled && backend!="device")
        {
#ifdef HORIZON_WITH_KOKKOS
            if(batch)
            {
                auto native_collision=manager.addNative("sdf_batch",batch->function());
                manager.addResidual("collision",{q,u},collision_scale*native_collision(q));
            }
            else
#endif
            {
            auto native_collision=manager.addNative("sdf",collision);
            auto centers=MX::vertcat({points(Slice(0,sphere_count)),points(Slice(count,count+sphere_count)),
                                     points(Slice(2*count,2*count+sphere_count))});
            manager.addResidual("collision",{q,u},collision_scale*native_collision(centers));
            }
        }
        horizon::IterativeLQR solver(manager,"dynamics",N,
            {{"ilqr.n_threads",threads},{"ilqr.constr_decomp_type",std::string("svd")},
             {"ilqr.huu_reg",damping/N},
             {"ilqr.kkt_decomp_type",decomposition},
             {"ilqr.constraint_violation_threshold",1e-7},{"ilqr.defect_norm_threshold",1e-7}},space);
        std::vector<int> running(N),all(N+1); std::iota(running.begin(),running.end(),0);std::iota(all.begin(),all.end(),0);
        solver.setResidual(running,manager,"effort"); solver.setResidual({N},manager,"target");
        solver.setConstraint(all,manager,"feet");
        if(collision_enabled && backend!="device") solver.setResidual(all,manager,"collision");
#ifdef HORIZON_WITH_KOKKOS
        if(batch)solver.addEvaluationBatch(*batch);
        if(device)solver.setBatchedCost(all,"collision",*device);
#endif
        Matrix lb=model->lowerPositionLimit.replicate(1,N+1),ub=model->upperPositionLimit.replicate(1,N+1);
        lb.col(0)=ub.col(0)=start; solver.setStateBounds(lb,ub);
        solver.setInputBounds(Matrix::Constant(model->nv,N,-1.),Matrix::Constant(model->nv,N,1.));
        const double setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-setup_start).count();
        std::ofstream report(output+".json");
        if(!report) throw std::runtime_error("cannot open report");
        report<<std::setprecision(12);
        report << "{\"model\":\"kinematic_configuration_velocity\",\"nodes\":"<<N
               <<",\"nq\":"<<model->nq<<",\"nv\":"<<model->nv<<",\"compiler\":"<<jsonString(__VERSION__)
               <<",\"collision_margin\":0.02,\"collision_smoothing\":0.005,\"collision_weight\":"<<collision_weight
               <<",\"urdf\":"<<jsonString(std::filesystem::absolute(urdf).string())<<",\"scenario\":"<<jsonString(scenario)
               <<",\"warmup_solves\":"<<warmup
#ifdef HORIZON_PROFILING
               <<",\"profiling_enabled\":true"
#else
               <<",\"profiling_enabled\":false"
#endif
               <<",\"backend\":\""<<backend<<"\",\"kokkos_threads\":"<<kokkos_threads
               <<",\"threads\":"<<threads<<",\"damping\":"<<damping<<",\"decomposition\":\""<<decomposition
               <<"\",\"spheres\":"<<sphere_count<<",\"voxel_size\":"<<voxel
               <<",\"collision_enabled\":"<<(collision_enabled?"true":"false")<<",\"setup_ms\":"<<setup_ms
               <<",\"self_pairs\":"<<self_pairs.size()
               <<",\"duration_s\":2,\"sdf_dimensions\":[151,151,151],\"sdf_origin\":[-3,-3,-3]"
               <<",\"obstacle_radius\":"<<obstacle_radius<<",\"obstacle_center\":["<<obstacle.x()<<','<<obstacle.y()<<','<<obstacle.z()
               <<"],\"target\":["<<target.x()<<','<<target.y()<<','<<target.z()<<"],\"samples\":[";
        bool valid=true;
        // Negative sample indices are warmup solves; every solve starts from the same guess.
        for(int sample=-warmup;sample<samples;++sample)
        {
            solver.setStateInitialGuess(start.replicate(1,N+1));solver.setInputInitialGuess(Matrix::Zero(model->nv,N));
            std::ofstream trace;
            Clock::time_point begin,iteration_begin;
            std::map<std::string,std::size_t> iteration_offsets;
            for(const auto* name:{"linearize_quadratize","backward_pass","line_search"})
            {
                const auto& timers=solver.getProfilingInfo().timings;auto it=timers.find(name);
                iteration_offsets[name]=it==timers.end()?0:it->second.size();
            }
            if(sample>=0)
            {
                trace.open(output+"_iterations_"+std::to_string(sample)+".csv");
                trace<<std::setprecision(12);
                trace<<"iteration,alpha,accepted,hxx_reg,cost,merit_derivative,step_length,defect,constraint,elapsed_ms,callback_interval_ms,linearize_quadratize_completed_since_callback_ms,backward_pass_completed_since_callback_ms,line_search_completed_since_callback_ms\n";
            }
            solver.setIterationCallback([&](const auto& r) {
                const auto now=Clock::now();
                if(trace.is_open())
                {
                    trace<<r.iter<<','<<r.alpha<<','<<r.accepted<<','<<r.hxx_reg<<','<<r.cost<<','
                         <<r.merit_der<<','<<r.step_length<<','<<r.defect_norm<<','<<r.constraint_violation<<','
                         <<std::chrono::duration<double,std::milli>(now-begin).count()<<','
                         <<std::chrono::duration<double,std::milli>(now-iteration_begin).count();
                    for(const auto* name:{"linearize_quadratize","backward_pass","line_search"})
                    {
                        const auto& timers=solver.getProfilingInfo().timings;auto it=timers.find(name);double us=0.;
                        if(it!=timers.end())
                        {us=std::accumulate(it->second.begin()+std::min(iteration_offsets[name],it->second.size()),it->second.end(),0.);iteration_offsets[name]=it->second.size();}
                        trace<<','<<us/1000.;
                    }
                    trace<<'\n';
                }
                iteration_begin=now;
                return true;
            });
            counters->values=0;counters->products=0;counters->nanos=0;
#ifdef HORIZON_WITH_KOKKOS
            if(batch)batch->resetStatistics();
            if(device)device->resetStatistics();
#endif
            const auto before=solver.getProfilingInfo().timings;
            const auto history_before=solver.getIterationHistory().size();
            begin=iteration_begin=Clock::now(); const bool converged=solver.solve(iterations);
            const double ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            auto value_calls=counters->values.load(),product_calls=counters->products.load(),collision_ns=counters->nanos.load();
#ifdef HORIZON_WITH_KOKKOS
            const auto batch_stats=batch?batch->statistics():CollisionBatchStatistics{};
            const auto device_stats=device?device->statistics():DeviceCollisionStatistics{};
            if(batch)
            {
                value_calls=batch_stats.value_batches*(N+1);product_calls=batch_stats.derivative_batches*(N+1);
                collision_ns=static_cast<unsigned long long>(batch_stats.backend_ms*1e6);
            }
            if(device)
            {
                value_calls=device_stats.value_batches*(N+1);product_calls=device_stats.derivative_batches*(N+1);
                collision_ns=0;
            }
#endif
            double drift=0,gap=0,min_clear=1e9,min_self=1e9;
            Vector actual(3*count),prediction(model->nq),defect(model->nv);
            for(int k=0;k<=N;++k)
            {
                fk->evaluate(solver.state(k),actual);
                for(int f=0;f<4;++f) drift=std::max(drift,(point(actual,sphere_count+f)-point(initial_points,sphere_count+f)).norm());
                for(int i=0;i<sphere_count;++i) min_clear=std::min(min_clear,field->query(point(actual,i)).distance-radii[i]);
                for(auto [i,j]:self_pairs)min_self=std::min(min_self,(point(actual,i)-point(actual,j)).norm()-radii[i]-radii[j]);
                if(k<N)
                {
                    space->integrate(solver.state(k),(2./N)*solver.input(k),prediction);
                    space->difference(prediction,solver.state(k+1),defect);gap=std::max(gap,defect.norm());
                }
            }
            const double error=(point(actual,count-1)-target).norm();
            valid=valid && error<.005 && drift<1e-4 && gap<1e-4 && (!collision_enabled || (min_clear>0 && min_self>0));
            if(sample>=0)valid=valid && converged;
            const auto& history=solver.getIterationHistory();
            const bool objective_available=history.size()>history_before;
            std::cout<<"sample="<<sample<<" solve_ms="<<ms<<" target_error="<<error
                     <<" foot_drift="<<drift<<" gap="<<gap<<" sdf_clearance="<<min_clear<<"\n";
            if(sample<0) continue;
            if(sample) report<<',';
            report<<"{\"solve_ms\":"<<ms<<",\"solver_converged\":"<<(converged?"true":"false")
                  <<",\"iterations\":"<<solver.getIterationHistory().size()-history_before<<",\"target_error\":"<<error
                  <<",\"objective\":";
            if(objective_available)report<<history.back().cost;else report<<"null";
            report
                  <<",\"foot_drift\":"<<drift<<",\"dynamics_gap\":"<<gap<<",\"min_sdf_clearance\":"<<min_clear
                  <<",\"min_self_clearance\":"<<min_self
                  <<",\"collision_value_calls\":"<<value_calls<<",\"collision_derivative_calls\":"<<product_calls
                  <<",\"sdf_queries\":"<<(value_calls+product_calls)*sphere_count
                  <<",\"self_queries\":"<<(value_calls+product_calls)*self_pairs.size()
                  <<",\"collision_cpu_ms\":"<<collision_ns*1e-6;
#ifdef HORIZON_WITH_KOKKOS
            if(batch)report<<",\"kokkos_execution_space\":\""<<KokkosCollisionBatch::executionSpace()
                <<"\",\"batch_value_passes\":"<<batch_stats.value_batches
                <<",\"batch_derivative_passes\":"<<batch_stats.derivative_batches
                <<",\"batch_kernel_launches\":"<<batch_stats.value_batches+2*batch_stats.derivative_batches
                <<",\"batch_node_evaluations\":"<<batch_stats.nodes
                <<",\"batch_prepare_ms\":"<<batch_stats.prepare_ms
                <<",\"batch_kinematics_ms\":"<<batch_stats.kinematics_ms
                <<",\"batch_backend_ms\":"<<batch_stats.backend_ms;
            if(device)report<<",\"device_execution_space\":\""<<DeviceCollisionCost::executionSpace()
                <<"\",\"device_prepare_ms\":"<<device_stats.prepare_ms
                <<",\"device_upload_ms\":"<<device_stats.upload_ms
                <<",\"device_kinematics_ms\":"<<device_stats.kinematics_ms
                <<",\"device_queries_ms\":"<<device_stats.queries_ms
                <<",\"device_accumulation_ms\":"<<device_stats.accumulation_ms
                <<",\"device_download_ms\":"<<device_stats.download_ms
                <<",\"device_upload_bytes\":"<<device_stats.upload_bytes
                <<",\"device_download_bytes\":"<<device_stats.download_bytes
                <<",\"device_peak_bytes\":"<<device_stats.device_bytes
                <<",\"device_kernel_launches\":"<<device_stats.kernel_launches
                <<",\"device_value_batches\":"<<device_stats.value_batches
                <<",\"device_derivative_batches\":"<<device_stats.derivative_batches;
#endif
            report<<",\"last_evaluated_cost_components\":{";
            bool cost_first=true;
            for(const auto& [name,values]:solver.getCostsValues())
            {
                double total=0.;bool present=false;
                for(int k=0;k<values.size();++k)if(std::isfinite(values[k]))
                {total+=values[k]*(k==N?1.:1./N);present=true;}
                if(!present)continue;if(!cost_first)report<<',';cost_first=false;
                report<<jsonString(name)<<':'<<total;
            }
            report<<"},\"phase_us\":{";
            bool first=true;
            for(const auto& [name,timings]:solver.getProfilingInfo().timings)
            {
                const auto it=before.find(name);const std::size_t offset=it==before.end()?0:it->second.size();
                if(!first) report<<',';first=false;
                report<<'"'<<name<<"\":"<<std::accumulate(timings.begin()+std::min(offset,timings.size()),timings.end(),0.);
            }
            report<<"},\"phase_statistics\":{";
            first=true;
            for(const auto& [name,timings]:solver.getProfilingInfo().timings)
            {
                const auto it=before.find(name);const std::size_t offset=it==before.end()?0:it->second.size();
                std::vector<double> values(timings.begin()+std::min(offset,timings.size()),timings.end());
                if(values.empty())continue;std::sort(values.begin(),values.end());
                const double total=std::accumulate(values.begin(),values.end(),0.);
                if(!first)report<<',';first=false;
                report<<jsonString(name)<<":{\"calls\":"<<values.size()<<",\"total_us\":"<<total
                      <<",\"mean_us\":"<<total/values.size()<<",\"min_us\":"<<values.front()
                      <<",\"p50_us\":"<<.5*(values[(values.size()-1)/2]+values[values.size()/2])
                      <<",\"p95_us\":"<<values[std::min(values.size()-1,std::size_t(std::ceil(.95*values.size())-1))]
                      <<",\"max_us\":"<<values.back()<<'}';
            }
            report<<"}}";
        }
        report<<"],\"valid\":"<<(valid?"true":"false")<<"}\n";
        std::ofstream trajectory(output+".csv");
        if(!trajectory) throw std::runtime_error("cannot open trajectory output");
        trajectory<<std::setprecision(12);
        trajectory<<"time";for(int i=0;i<model->nq;++i) trajectory<<",q"<<i;
        for(int i=0;i<model->nv;++i) trajectory<<",u"<<i;trajectory<<'\n';
        for(int k=0;k<=N;++k)
        {
            trajectory<<2.*k/N;for(int i=0;i<model->nq;++i) trajectory<<','<<solver.state(k)[i];
            for(int i=0;i<model->nv;++i) trajectory<<','<<(k<N?solver.input(k)[i]:0.);trajectory<<'\n';
        }
        return valid?0:1;
    }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
}
