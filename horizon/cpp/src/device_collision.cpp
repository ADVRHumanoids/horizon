#include "horizon/device_collision.h"
#include <Kokkos_Core.hpp>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <set>

namespace horizon::collision
{
namespace
{
using Space=Kokkos::DefaultExecutionSpace;
using Memory=Space::memory_space;
using V2=Kokkos::View<double**,Kokkos::LayoutRight,Memory>;
using V3=Kokkos::View<double***,Kokkos::LayoutRight,Memory>;
using R2=Kokkos::View<double**,Kokkos::LayoutLeft,Memory>;
using Clock=std::chrono::steady_clock;
double milliseconds(Clock::time_point start)
{return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}

// One pair of events per workspace; timings include transfers on the default
// execution stream. The portable path uses fenced wall time.
struct StageTimer
{
#ifdef KOKKOS_ENABLE_CUDA
    cudaEvent_t begin{},end{};
    static void check(cudaError_t status)
    {if(status!=cudaSuccess)throw std::runtime_error(std::string("collision CUDA timing: ")+cudaGetErrorString(status));}
    StageTimer()
    {
        check(cudaEventCreate(&begin));
        const auto status=cudaEventCreate(&end);
        if(status!=cudaSuccess){cudaEventDestroy(begin);check(status);}
    }
    ~StageTimer(){cudaEventDestroy(begin);cudaEventDestroy(end);}
    void start(){check(cudaEventRecord(begin,Space().cuda_stream()));}
    double stop()
    {
        check(cudaEventRecord(end,Space().cuda_stream()));check(cudaEventSynchronize(end));
        float ms=0;check(cudaEventElapsedTime(&ms,begin,end));return ms;
    }
#else
    Clock::time_point begin;
    void start(){Space().fence();begin=Clock::now();}
    double stop(){Space().fence();return milliseconds(begin);}
#endif
};

KOKKOS_INLINE_FUNCTION void identity(double* p)
{for(int i=0;i<12;++i)p[i]=0.;p[0]=p[4]=p[8]=1.;}
KOKKOS_INLINE_FUNCTION void compose(const double* a,const double* b,double* out)
{
    for(int i=0;i<3;++i)
    {
        out[9+i]=a[9+i];
        for(int k=0;k<3;++k)out[9+i]+=a[3*i+k]*b[9+k];
        for(int j=0;j<3;++j){out[3*i+j]=0.;for(int k=0;k<3;++k)out[3*i+j]+=a[3*i+k]*b[3*k+j];}
    }
}
KOKKOS_INLINE_FUNCTION void act(const double* pose,const double* p,double* out)
{for(int a=0;a<3;++a){out[a]=pose[9+a];for(int b=0;b<3;++b)out[a]+=pose[3*a+b]*p[b];}}

// Add g^T dp/dq for one sphere. Every invocation owns one residual row;
// common ancestors cancel naturally for self-collision, without atomics.
template<class JV,class TV,class SV>
KOKKOS_INLINE_FUNCTION void project(const JV& joints,const TV& transforms,const SV& spheres,
    const V3& centers,const V3& jac,int node,int row,int sphere,const double* g,double sign)
{
    int joint=spheres(sphere).joint;
    while(joint)
    {
        const auto data=joints(joint);double pose[12];
        for(int a=0;a<12;++a)pose[a]=transforms(node,joint,a);
        double relative[3],cross[3];
        for(int a=0;a<3;++a)relative[a]=centers(node,sphere,a)-pose[9+a];
        cross[0]=relative[1]*g[2]-relative[2]*g[1];
        cross[1]=relative[2]*g[0]-relative[0]*g[2];
        cross[2]=relative[0]*g[1]-relative[1]*g[0];
        if(data.type==DeviceRobotModel::FreeFlyer)
        {
            for(int col=0;col<3;++col)for(int a=0;a<3;++a)
            {
                jac(node,row,data.v+col)+=sign*g[a]*pose[3*a+col];
                jac(node,row,data.v+3+col)+=sign*cross[a]*pose[3*a+col];
            }
        }
        else if(data.type!=DeviceRobotModel::Fixed)
        {
            double result=0.;
            for(int a=0;a<3;++a)
            {
                double axis=0.;for(int b=0;b<3;++b)axis+=pose[3*a+b]*data.axis[b];
                result+=axis*(data.type==DeviceRobotModel::Prismatic?g[a]:cross[a]);
            }
            jac(node,row,data.v)+=sign*result;
        }
        joint=data.parent;
    }
}
struct SharedStatistics {std::mutex mutex;DeviceCollisionStatistics value;};
}

struct DeviceCollisionCost::Configuration
{
    DeviceRobotModel model;
    std::shared_ptr<const SdfGrid> grid;
    std::vector<SphereDistanceFunction::Pair> pairs;
    double weight,margin,smoothing;
    int state_size,offset;
    std::shared_ptr<SharedStatistics> statistics=std::make_shared<SharedStatistics>();
    int outputs() const{return (grid?int(model.spheres.size()):0)+int(pairs.size());}
};

struct DeviceCollisionCost::Impl
{
    Kokkos::View<DeviceRobotModel::Joint*,Memory> joints;
    Kokkos::View<DeviceRobotModel::Sphere*,Memory> spheres;
    Kokkos::View<int**,Kokkos::LayoutRight,Memory> pairs;
    Kokkos::View<float*,Memory> grid;
    Kokkos::View<int,Memory> error;
    V2 q;
    R2 result;
    V3 transforms,centers,jac;
    V2::HostMirror host_q;
    R2::HostMirror host_result;
    ilqr::Vector values;
    ilqr::Matrix gradients,hessians;
    StageTimer timer;
    int nodes=0,tile=0;
    bool ready=false,derivatives=false;
    std::uint64_t bytes=0;
    explicit Impl(const Configuration& c)
    {
        if(!Kokkos::is_initialized())throw std::logic_error("initialize Kokkos before creating a device collision cost");
        joints=decltype(joints)("robot joints",c.model.joints.size());
        spheres=decltype(spheres)("local spheres",c.model.spheres.size());
        pairs=decltype(pairs)("self pairs",c.pairs.size(),2);
        grid=decltype(grid)("sdf",c.grid?c.grid->values().size():0);
        auto hj=Kokkos::create_mirror_view(joints);auto hs=Kokkos::create_mirror_view(spheres);
        auto hp=Kokkos::create_mirror_view(pairs);auto hg=Kokkos::create_mirror_view(grid);
        for(std::size_t i=0;i<c.model.joints.size();++i)hj(i)=c.model.joints[i];
        for(std::size_t i=0;i<c.model.spheres.size();++i)hs(i)=c.model.spheres[i];
        for(std::size_t i=0;i<c.pairs.size();++i){hp(i,0)=c.pairs[i].first;hp(i,1)=c.pairs[i].second;}
        if(c.grid)for(std::size_t i=0;i<c.grid->values().size();++i)hg(i)=c.grid->values()[i];
        Kokkos::deep_copy(joints,hj);Kokkos::deep_copy(spheres,hs);
        Kokkos::deep_copy(pairs,hp);Kokkos::deep_copy(grid,hg);
        error=decltype(error)("collision status");
    }
    void resize(int n,const Configuration& c,bool deriv)
    {
        const int nv=c.model.nv,packed=nv*(nv+1)/2;
        if(nodes!=n)
        {
            nodes=n;q=V2("configurations",n,c.model.nq);host_q=Kokkos::create_mirror_view(q);
            transforms=V3("joint transforms",n,c.model.joints.size(),12);
            centers=V3("world centers",n,c.model.spheres.size(),3);
            values.resize(n);jac=V3();result=R2();
        }
        const int columns=deriv?1+nv+packed:1;
        if(result.extent(1)<std::size_t(columns))
        {result=R2("cost contributions",n,columns);host_result=Kokkos::create_mirror_view(result);}
        if(deriv && !jac.extent(0))
        {
            const std::size_t rows=(64u*1024u*1024u)/(std::size_t(n)*(nv+1)*sizeof(double));
            if(rows<1)throw std::invalid_argument("one collision query tile exceeds the 64 MiB workspace budget");
            tile=std::min<std::size_t>(c.outputs(),rows);
            jac=V3("resident residual Jacobians",n,tile,nv+1);
            gradients.resize(nv,n);hessians.resize(packed,n);
        }
        // Value-only passes retain an existing derivative workspace for reuse.
        bytes=joints.span()*sizeof(DeviceRobotModel::Joint)+spheres.span()*sizeof(DeviceRobotModel::Sphere)+
            pairs.span()*sizeof(int)+grid.span()*sizeof(float)+sizeof(int)+
            (q.span()+transforms.span()+centers.span()+jac.span()+result.span())*sizeof(double);
    }
    void check(int node,bool deriv) const
    {if(!ready || node<0 || node>=nodes || (deriv && !derivatives))throw std::logic_error("device collision result unavailable for this pass");}
};

DeviceCollisionCost::DeviceCollisionCost(DeviceRobotModel model,std::shared_ptr<const SdfGrid> grid,
    std::vector<SphereDistanceFunction::Pair> pairs,double weight,double margin,double smoothing,int state_size,int offset)
{
    if(!Kokkos::is_initialized())throw std::logic_error("initialize Kokkos before creating a device collision cost");
    if(model.nq<1 || model.nv<1 || model.nv>1024 || model.joints.empty() || offset<0 ||
       !std::isfinite(weight) || weight<0)throw std::invalid_argument("invalid device robot/cost dimensions or weight");
    if(!state_size)state_size=model.nq;
    if(state_size<model.nq || offset>state_size-model.nq)throw std::invalid_argument("configuration block exceeds the stored state");
    std::vector<double> radii;
    std::set<int> q_indices,v_indices;
    for(std::size_t i=0;i<model.joints.size();++i)
    {
        const auto& j=model.joints[i];
        if((i && (j.parent<0 || j.parent>=int(i))) || (!i && (j.parent || j.type!=DeviceRobotModel::Fixed)) ||
           j.type<DeviceRobotModel::Fixed || j.type>DeviceRobotModel::Continuous)
            throw std::invalid_argument("invalid device robot topology");
        const int nq=j.type==DeviceRobotModel::Fixed?0:(j.type==DeviceRobotModel::FreeFlyer?7:(j.type==DeviceRobotModel::Continuous?2:1));
        const int nv=j.type==DeviceRobotModel::Fixed?0:(j.type==DeviceRobotModel::FreeFlyer?6:1);
        if(j.q<0 || j.v<0 || j.q>model.nq-nq || j.v>model.nv-nv)throw std::invalid_argument("invalid device joint indices");
        for(int a=0;a<nq;++a)if(!q_indices.insert(j.q+a).second)throw std::invalid_argument("overlapping device configuration indices");
        for(int a=0;a<nv;++a)if(!v_indices.insert(j.v+a).second)throw std::invalid_argument("overlapping device tangent indices");
        for(double x:j.placement)if(!std::isfinite(x))throw std::invalid_argument("nonfinite joint placement");
        double norm=0.;for(double x:j.axis){if(!std::isfinite(x))throw std::invalid_argument("nonfinite joint axis");norm+=x*x;}
        if(nv==1 && std::abs(norm-1.)>1e-10)throw std::invalid_argument("device joint axis must be normalized");
    }
    if(q_indices.size()!=std::size_t(model.nq) || v_indices.size()!=std::size_t(model.nv))throw std::invalid_argument("incomplete device joint indexing");
    for(const auto& s:model.spheres)
    {
        if(s.joint<0 || s.joint>=int(model.joints.size()))throw std::invalid_argument("invalid sphere parent");
        for(double x:s.center)if(!std::isfinite(x))throw std::invalid_argument("nonfinite sphere center");
        radii.push_back(s.radius);
    }
    std::vector<std::shared_ptr<const DistanceField>> fields;if(grid)fields.push_back(grid);
    SphereDistanceFunction validate(radii,fields,pairs,margin,smoothing);
    auto c=std::make_shared<Configuration>();c->model=std::move(model);c->grid=std::move(grid);c->pairs=std::move(pairs);
    c->weight=weight;c->margin=margin;c->smoothing=smoothing;c->state_size=state_size;c->offset=offset;
    _configuration=c;_impl=std::make_unique<Impl>(*c);
}
DeviceCollisionCost::DeviceCollisionCost(std::shared_ptr<const Configuration> c):
    _configuration(std::move(c)),_impl(std::make_unique<Impl>(*_configuration)){}
DeviceCollisionCost::~DeviceCollisionCost()=default;
std::unique_ptr<ilqr::BatchedCost> DeviceCollisionCost::clone() const
{return std::unique_ptr<ilqr::BatchedCost>(new DeviceCollisionCost(_configuration));}
int DeviceCollisionCost::stateSize() const{return _configuration->state_size;}
int DeviceCollisionCost::tangentSize() const{return _configuration->model.nv;}
const char* DeviceCollisionCost::executionSpace(){return Space::name();}
double DeviceCollisionCost::value(int node) const{_impl->check(node,false);return _impl->values[node];}
ilqr::Input DeviceCollisionCost::gradient(int node) const{_impl->check(node,true);return _impl->gradients.col(node);}
ilqr::Input DeviceCollisionCost::packedHessian(int node) const{_impl->check(node,true);return _impl->hessians.col(node);}
DeviceCollisionStatistics DeviceCollisionCost::statistics() const
{std::lock_guard lock(_configuration->statistics->mutex);return _configuration->statistics->value;}
void DeviceCollisionCost::resetStatistics()
{std::lock_guard lock(_configuration->statistics->mutex);_configuration->statistics->value={};}

void DeviceCollisionCost::prepare(ilqr::MatrixInput states,bool derivatives)
{
    const auto start=Clock::now();const auto& c=*_configuration;auto& w=*_impl;
    w.ready=false;
    const int n=states.cols(),nq=c.model.nq,nv=c.model.nv,ns=c.model.spheres.size(),nj=c.model.joints.size();
    const int outputs=c.outputs(),packed=nv*(nv+1)/2;
    if(n<1 || states.rows()!=c.state_size || !states.allFinite() ||
       std::uint64_t(n)*std::max(outputs,1+nv+packed)>std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid device collision trajectory dimensions or values");
    // Pinocchio's configuration manifold assumes normalized unit quaternions.
    for(const auto& joint:c.model.joints)if(joint.type==DeviceRobotModel::FreeFlyer || joint.type==DeviceRobotModel::Continuous)
    {
        const int count=joint.type==DeviceRobotModel::FreeFlyer?4:2;
        const int offset=c.offset+joint.q+(count==4?3:0);
        for(int node=0;node<n;++node)if(std::abs(states.col(node).segment(offset,count).squaredNorm()-1.)>1e-6)
            throw ilqr::EvaluationError("device collision requires normalized joint configurations");
    }
    w.resize(n,c,derivatives);DeviceCollisionStatistics stat;
    for(int node=0;node<n;++node)for(int a=0;a<nq;++a)w.host_q(node,a)=states(c.offset+a,node);
    const int result_columns=derivatives?1+nv+packed:1;
    auto active_result=Kokkos::subview(w.result,Kokkos::ALL(),std::make_pair(0,result_columns));
    auto active_host_result=Kokkos::subview(w.host_result,Kokkos::ALL(),std::make_pair(0,result_columns));
    w.timer.start();Kokkos::deep_copy(w.q,w.host_q);Kokkos::deep_copy(w.error,0);Kokkos::deep_copy(active_result,0.);
    stat.upload_ms=w.timer.stop();
    const auto q=w.q;const auto joints=w.joints;const auto spheres=w.spheres;
    const auto transforms=w.transforms;const auto centers=w.centers;const auto error=w.error;
    w.timer.start();
    Kokkos::parallel_for("collision device forward kinematics",Kokkos::RangePolicy<Space>(0,n),KOKKOS_LAMBDA(int node)
    {
        for(int joint=0;joint<nj;++joint)
        {
            const auto data=joints(joint);double motion[12],placed[12],parent[12],world[12];identity(motion);
            if(data.type==DeviceRobotModel::FreeFlyer)
            {
                const double x=q(node,data.q+3),y=q(node,data.q+4),z=q(node,data.q+5),t=q(node,data.q+6);
                motion[0]=1-2*(y*y+z*z);motion[1]=2*(x*y-z*t);motion[2]=2*(x*z+y*t);
                motion[3]=2*(x*y+z*t);motion[4]=1-2*(x*x+z*z);motion[5]=2*(y*z-x*t);
                motion[6]=2*(x*z-y*t);motion[7]=2*(y*z+x*t);motion[8]=1-2*(x*x+y*y);
                for(int a=0;a<3;++a)motion[9+a]=q(node,data.q+a);
            }
            else if(data.type==DeviceRobotModel::Revolute || data.type==DeviceRobotModel::Continuous)
            {
                const double cosine=data.type==DeviceRobotModel::Continuous?q(node,data.q):Kokkos::cos(q(node,data.q));
                const double sine=data.type==DeviceRobotModel::Continuous?q(node,data.q+1):Kokkos::sin(q(node,data.q));
                for(int a=0;a<3;++a)for(int b=0;b<3;++b)
                    motion[3*a+b]=(a==b?cosine:0.)+(1-cosine)*data.axis[a]*data.axis[b];
                motion[1]-=sine*data.axis[2];motion[2]+=sine*data.axis[1];
                motion[3]+=sine*data.axis[2];motion[5]-=sine*data.axis[0];
                motion[6]-=sine*data.axis[1];motion[7]+=sine*data.axis[0];
            }
            else if(data.type==DeviceRobotModel::Prismatic)
                for(int a=0;a<3;++a)motion[9+a]=q(node,data.q)*data.axis[a];
            compose(data.placement,motion,placed);
            if(joint){for(int a=0;a<12;++a)parent[a]=transforms(node,data.parent,a);compose(parent,placed,world);}
            else identity(world);
            for(int a=0;a<12;++a)transforms(node,joint,a)=world[a];
        }
        for(int sphere=0;sphere<ns;++sphere)
        {
            double pose[12],p[3];for(int a=0;a<12;++a)pose[a]=transforms(node,spheres(sphere).joint,a);
            act(pose,spheres(sphere).center,p);for(int a=0;a<3;++a)centers(node,sphere,a)=p[a];
        }
    });
    stat.kinematics_ms=w.timer.stop();++stat.kernel_launches;
    const auto grid=w.grid;const auto pairs=w.pairs;const auto jac=w.jac;const auto result=w.result;
    const bool environment=bool(c.grid);const int nx=environment?c.grid->dimensions().x():0,ny=environment?c.grid->dimensions().y():0,nz=environment?c.grid->dimensions().z():0;
    const double ox=environment?c.grid->origin().x():0.,oy=environment?c.grid->origin().y():0.,oz=environment?c.grid->origin().z():0.;
    const double voxel=environment?c.grid->voxelSize():1.,margin=c.margin,smoothing=c.smoothing,weight=c.weight;
    const int tile=derivatives?w.tile:outputs;
    for(int first=0;first<outputs;first+=tile)
    {
        const int rows=std::min(tile,outputs-first);w.timer.start();
        Kokkos::parallel_for("collision device residuals and tangent derivatives",Kokkos::RangePolicy<Space>(0,n*rows),KOKKOS_LAMBDA(int index)
        {
            const int node=index/rows,row=index%rows,query=first+row;
            if(derivatives)for(int col=0;col<=nv;++col)jac(node,row,col)=0.;
            double clear=0.,g[3]={0.,0.,0.};int i=0,other=0;
            const bool env=environment && query<ns;
            if(env)
            {
                i=query;const double p[3]={(centers(node,i,0)-ox)/voxel,(centers(node,i,1)-oy)/voxel,(centers(node,i,2)-oz)/voxel};
                if(!Kokkos::isfinite(p[0]) || !Kokkos::isfinite(p[1]) || !Kokkos::isfinite(p[2]) ||
                   p[0]<0 || p[1]<0 || p[2]<0 || p[0]>nx-1 || p[1]>ny-1 || p[2]>nz-1)
                {Kokkos::atomic_exchange(&error(),1);return;}
                const int dims[3]={nx,ny,nz};int lo[3];double f[3];
                for(int a=0;a<3;++a){lo[a]=int(Kokkos::floor(p[a]));if(lo[a]>dims[a]-2)lo[a]=dims[a]-2;f[a]=p[a]-lo[a];}
                for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x)
                {
                    const double wx=x?f[0]:1-f[0],wy=y?f[1]:1-f[1],wz=z?f[2]:1-f[2];
                    const double v=grid((std::size_t(lo[2]+z)*ny+lo[1]+y)*nx+lo[0]+x);
                    clear+=wx*wy*wz*v;g[0]+=(x?1.:-1.)*wy*wz*v/voxel;
                    g[1]+=(y?1.:-1.)*wx*wz*v/voxel;g[2]+=(z?1.:-1.)*wx*wy*v/voxel;
                }
                clear-=spheres(i).radius;
            }
            else
            {
                const int pair=query-(environment?ns:0);i=pairs(pair,0);other=pairs(pair,1);
                double norm=1e-24;
                for(int a=0;a<3;++a){g[a]=centers(node,i,a)-centers(node,other,a);norm+=g[a]*g[a];}
                const double length=Kokkos::sqrt(norm);clear=length-spheres(i).radius-spheres(other).radius;
                for(int a=0;a<3;++a)g[a]/=length;
            }
            const double z=(margin-clear)/smoothing;
            const double residual=smoothing*(z>40?z:(z< -40?Kokkos::exp(z):Kokkos::log1p(Kokkos::exp(z))));
            if(!Kokkos::isfinite(residual)){Kokkos::atomic_exchange(&error(),1);return;}
            if(!derivatives){Kokkos::atomic_add(&result(node,0),.5*weight*residual*residual);return;}
            const double e=Kokkos::exp(z>=0?-Kokkos::fmin(z,700.):Kokkos::fmax(z,-700.));
            const double scale=z>=0?-1./(1.+e):-e/(1.+e);
            for(int a=0;a<3;++a)g[a]*=scale;
            jac(node,row,nv)=residual;
            project(joints,transforms,spheres,centers,jac,node,row,i,g,1.);
            if(!env)project(joints,transforms,spheres,centers,jac,node,row,other,g,-1.);
        });
        stat.queries_ms+=w.timer.stop();++stat.kernel_launches;
        if(derivatives)
        {
            w.timer.start();
            Kokkos::parallel_for("collision device Gauss Newton reduction",Kokkos::RangePolicy<Space>(0,n*(1+nv+packed)),KOKKOS_LAMBDA(int index)
            {
                const int columns=1+nv+packed,node=index/columns,entry=index%columns;
                int a=nv,b=nv;double scale=weight;
                if(!entry)scale*=.5;
                else if(entry<=nv)a=entry-1;
                else
                {
                    int k=entry-1-nv;a=0;
                    while(k>=nv-a){k-=nv-a;++a;}b=a+k;
                }
                double sum=0.;for(int row=0;row<rows;++row)sum+=jac(node,row,a)*jac(node,row,b);
                result(node,entry)+=scale*sum;
            });
            stat.accumulation_ms+=w.timer.stop();++stat.kernel_launches;
        }
    }
    w.timer.start();Kokkos::deep_copy(active_host_result,active_result);int failed=0;Kokkos::deep_copy(failed,w.error);
    stat.download_ms=w.timer.stop();
    if(failed)throw ilqr::EvaluationError("device collision query failed (out of grid or nonfinite)");
    for(int node=0;node<n;++node)
    {
        w.values[node]=w.host_result(node,0);
        if(derivatives)
        {
            for(int a=0;a<nv;++a)w.gradients(a,node)=w.host_result(node,1+a);
            for(int a=0;a<packed;++a)w.hessians(a,node)=w.host_result(node,1+nv+a);
        }
    }
    if(!w.values.allFinite() || (derivatives && (!w.gradients.allFinite() || !w.hessians.allFinite())))
        throw ilqr::EvaluationError("nonfinite device collision contribution");
    w.derivatives=derivatives;w.ready=true;
    stat.nodes=n;stat.value_batches=derivatives?0:1;stat.derivative_batches=derivatives?1:0;
    stat.sdf_queries=environment?std::uint64_t(n)*ns:0;stat.self_queries=std::uint64_t(n)*c.pairs.size();
    stat.upload_bytes=std::uint64_t(n)*nq*sizeof(double);stat.download_bytes=std::uint64_t(n)*(derivatives?1+nv+packed:1)*sizeof(double)+sizeof(int);
    stat.device_bytes=w.bytes;stat.prepare_ms=milliseconds(start);
    std::lock_guard lock(c.statistics->mutex);auto& s=c.statistics->value;
    s.value_batches+=stat.value_batches;s.derivative_batches+=stat.derivative_batches;s.nodes+=stat.nodes;
    s.sdf_queries+=stat.sdf_queries;s.self_queries+=stat.self_queries;s.upload_bytes+=stat.upload_bytes;
    s.download_bytes+=stat.download_bytes;s.kernel_launches+=stat.kernel_launches;s.device_bytes=std::max(s.device_bytes,stat.device_bytes);
    s.prepare_ms+=stat.prepare_ms;s.upload_ms+=stat.upload_ms;s.kinematics_ms+=stat.kinematics_ms;
    s.queries_ms+=stat.queries_ms;s.accumulation_ms+=stat.accumulation_ms;s.download_ms+=stat.download_ms;
}
}
