// Isolated sparse/dense distance-gradient and incremental ESDF integration probe.
#include <nvblox/map/layer.h>
#include <nvblox/map/common_names.h>
#include <nvblox/map/voxels.h>
#include <nvblox/integrators/esdf_integrator.h>
#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>
#include <nvblox/core/unified_vector.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <cuda_fp16.h>
#ifdef HORIZON_CUROBO_SCENE
#include <curobo_robot_runtime/scene.h>
#endif

using namespace nvblox;
constexpr int dim=151;
constexpr float voxel=.04f;
struct Point {double x,y,z;};
struct Sample {double distance,gx,gy,gz;int valid;};
bool check_only=false;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}

__global__ void exportDense(Index3DDeviceHashMapType<EsdfBlock> hash,float* fp32,__half* fp16,int* unknown)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dim*dim*dim)return;
    const int x=i%dim,y=(i/dim)%dim,z=i/(dim*dim);EsdfVoxel* ptr=nullptr;
    float value=0;
    if(!getVoxelPtr(hash,Index3D(x/8,y/8,z/8),Index3D(x%8,y%8,z%8),&ptr) || !ptr->observed)
        atomicAdd(unknown,1);
    else value=voxel*sqrtf(ptr->squared_distance_vox)*(ptr->is_inside?-1.f:1.f);
    fp32[i]=value;
    if(fp16)fp16[(x*dim+y)*dim+z]=__float2half(value);
}

template<bool sparse,bool esdf=false>
__global__ void query(int n,const Point* points,const float* dense,
    Index3DDeviceHashMapType<TsdfBlock> tsdf_hash,
    Index3DDeviceHashMapType<EsdfBlock> esdf_hash,Sample* samples)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const Point p=points[i];const double s[3]={(p.x+3.)/voxel,(p.y+3.)/voxel,(p.z+3.)/voxel};
    int lo[3];double f[3];
    samples[i]={0,0,0,0,0};
    for(int a=0;a<3;++a){if(s[a]<0 || s[a]>dim-1)return;lo[a]=min(int(floor(s[a])),dim-2);f[a]=s[a]-lo[a];}
    double value=0.,g[3]={0,0,0};
    for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x)
    {
        const int ix=lo[0]+x,iy=lo[1]+y,iz=lo[2]+z;double v;
        if constexpr(sparse)
        {
            const Index3D block(ix/8,iy/8,iz/8),index(ix%8,iy%8,iz%8);
            if constexpr(esdf)
            {
                EsdfVoxel* ptr=nullptr;
                if(!getVoxelPtr(esdf_hash,block,index,&ptr) || !ptr->observed)return;
                v=voxel*sqrt(double(ptr->squared_distance_vox))*(ptr->is_inside?-1.:1.);
            }
            else
            {
                TsdfVoxel* ptr=nullptr;
                if(!getVoxelPtr(tsdf_hash,block,index,&ptr) || ptr->weight<=0)return;
                v=ptr->distance;
            }
        }
        else v=dense[(iz*dim+iy)*dim+ix];
        const double wx=x?f[0]:1-f[0],wy=y?f[1]:1-f[1],wz=z?f[2]:1-f[2];
        value+=wx*wy*wz*v;g[0]+=(x?1.:-1.)*wy*wz*v/voxel;
        g[1]+=(y?1.:-1.)*wx*wz*v/voxel;g[2]+=(z?1.:-1.)*wx*wy*v/voxel;
    }
    samples[i]={value,g[0],g[1],g[2],1};
}

template<class Fn> std::pair<double,double> timing(Fn fn)
{
    for(int i=0;i<(check_only?1:5);++i)fn();check(cudaDeviceSynchronize());
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    std::vector<double> samples;
    const int count=check_only?1:20,repeats=check_only?1:10;
    for(int sample=0;sample<count;++sample)
    {
        check(cudaEventRecord(begin));for(int repeat=0;repeat<repeats;++repeat)fn();
        check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms=0;
        check(cudaEventElapsedTime(&ms,begin,end));samples.push_back(ms/repeats);
    }
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));std::sort(samples.begin(),samples.end());
    return {(.5*(samples[(count-1)/2]+samples[count/2])),samples[int(std::ceil(.95*count))-1]};
}

int main(int argc,char** argv)
{
    try
    {
        std::cout<<std::setprecision(10)<<std::unitbuf;
        if(argc>1){if(std::string(argv[1])!="--check-only")throw std::invalid_argument("expected --check-only");check_only=true;}
        DefaultStream stream(nullptr);
        BlockMemoryPoolParams host_pool(MemoryType::kHost);host_pool.num_preallocated_blocks=7000;
        BlockMemoryPoolParams device_pool(MemoryType::kDevice);device_pool.num_preallocated_blocks=7000;
        TsdfLayer host(voxel,host_pool),device(voxel,device_pool);
        std::vector<float> grid(dim*dim*dim);
        for(int z=0;z<dim;++z)for(int y=0;y<dim;++y)for(int x=0;x<dim;++x)
        {
            const float px=-3+x*voxel,py=-3+y*voxel,pz=-3+z*voxel;
            const float d=std::sqrt(px*px+py*py+pz*pz)-.5f;
            grid[(z*dim+y)*dim+x]=d;
            auto block=host.allocateBlockAtIndex(Index3D(x/8,y/8,z/8));
            auto& v=block->voxels[x%8][y%8][z%8];v.distance=d;v.weight=1.f;
        }
        const auto begin=std::chrono::steady_clock::now();device.copyFromAsync(host,stream);stream.synchronize();
        auto& view=device.getGpuLayerView(stream);stream.synchronize();
        const double upload_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        const auto hash=view.getHash().impl_;
        device_vector<float> dense;dense.copyFromAsync(grid,stream);stream.synchronize();
        const auto used_blocks=device.getAllBlockIndices().size();
        std::cout<<"{\"blocks\":"<<used_blocks<<",\"allocated_pool_blocks\":"<<device.numAllocatedBlocks()
                 <<",\"dense_payload_bytes\":"<<grid.size()*sizeof(float)
                 <<",\"sparse_voxel_payload_bytes\":"<<used_blocks*512*sizeof(TsdfVoxel)
                 <<",\"sparse_pool_bytes\":"<<device.numAllocatedBlocks()*512*sizeof(TsdfVoxel)
                 <<",\"map_upload_and_hash_ms\":"<<upload_ms<<",\"cases\":[";
        std::mt19937 rng(42);std::uniform_real_distribution<double> random(-2.9,2.9);
        bool first=true;
        const std::vector<int> counts=check_only?std::vector<int>{31*94}:std::vector<int>{31*94,257*94,1025*94,1000000};
        for(int n:counts)
        {
            std::vector<Point> points(n);for(auto& p:points)p={random(rng),random(rng),random(rng)};
            points.back()={100,0,0}; // Unknown/outside must report invalid.
            device_vector<Point> dp;dp.copyFromAsync(points,stream);
            device_vector<Sample> a(n),b(n);stream.synchronize();
            auto dense_launch=[&]{query<false><<<(n+255)/256,256>>>(n,dp.data(),dense.data(),hash,{},a.data());};
            auto sparse_launch=[&]{query<true><<<(n+255)/256,256>>>(n,dp.data(),nullptr,hash,{},b.data());};
            const auto dense_time=timing(dense_launch),sparse_time=timing(sparse_launch);
            const auto ha=a.toVectorAsync(stream),hb=b.toVectorAsync(stream);stream.synchronize();
            double distance_error=0,gradient_error=0;int valid=0;
            for(int i=0;i<n;++i)
            {
                if(ha[i].valid!=hb[i].valid)throw std::runtime_error("sparse/dense validity mismatch");
                if(!ha[i].valid)continue;++valid;
                distance_error=std::max(distance_error,std::abs(ha[i].distance-hb[i].distance));
                gradient_error=std::max({gradient_error,std::abs(ha[i].gx-hb[i].gx),std::abs(ha[i].gy-hb[i].gy),std::abs(ha[i].gz-hb[i].gz)});
            }
            if(distance_error>1e-10 || gradient_error>1e-10 || valid!=n-1)throw std::runtime_error("sparse/dense numerical parity failed");
            if(!first)std::cout<<',';first=false;
            std::cout<<"{\"queries\":"<<n<<",\"dense_median_ms\":"<<dense_time.first<<",\"dense_p95_ms\":"<<dense_time.second
                     <<",\"sparse_median_ms\":"<<sparse_time.first<<",\"sparse_p95_ms\":"<<sparse_time.second
                     <<",\"distance_max_error\":"<<distance_error<<",\"gradient_max_error\":"<<gradient_error<<",\"invalid_queries\":"<<n-valid<<'}';
        }
        // An unobserved interpolation corner must not silently become free space.
        auto hb=host.getBlockAtIndex(Index3D(9,9,9));auto db=device.getBlockAtIndex(Index3D(9,9,9));
        hb->voxels[3][3][3].weight=0.f;
        check(cudaMemcpy(db.get(),hb.get(),sizeof(TsdfBlock),cudaMemcpyHostToDevice));
        device_vector<Point> missing;missing.copyFromAsync(std::vector<Point>{{.005,.005,.005}},stream);
        device_vector<Sample> missing_result(1);stream.synchronize();
        query<true><<<1,1>>>(1,missing.data(),nullptr,hash,{},missing_result.data());
        const auto invalid=missing_result.toVectorAsync(stream);stream.synchronize();
        if(invalid[0].valid)throw std::runtime_error("unobserved corner incorrectly treated as valid");
        hb->voxels[3][3][3].weight=1.f;check(cudaMemcpy(db.get(),hb.get(),sizeof(TsdfBlock),cudaMemcpyHostToDevice));
        // Build an actual ESDF through nvblox, rather than merely importing SDF samples.
        EsdfLayer esdf(voxel,device_pool);EsdfIntegrator integrator(std::make_shared<DefaultStream>(nullptr));integrator.max_esdf_distance_m(5.f);
        const auto blocks=device.getAllBlockIndices();
        auto esdf_begin=std::chrono::steady_clock::now();integrator.integrateBlocks(device,blocks,&esdf);check(cudaDeviceSynchronize());
        double first_esdf_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-esdf_begin).count();
        std::vector<double> updates;
        for(int i=0;i<5;++i)
        {esdf_begin=std::chrono::steady_clock::now();integrator.integrateBlocks(device,blocks,&esdf);check(cudaDeviceSynchronize());updates.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-esdf_begin).count());}
        std::sort(updates.begin(),updates.end());
        auto& ev=esdf.getGpuLayerView(stream);stream.synchronize();
        std::vector<Point> near;for(int i=0;i<1000;++i){const double x=-1.+2.*i/999;near.push_back({x,.13,.17});}
        device_vector<Point> dp;dp.copyFromAsync(near,stream);device_vector<Sample> output(near.size());stream.synchronize();
        query<true,true><<<(near.size()+255)/256,256>>>(near.size(),dp.data(),nullptr,hash,ev.getHash().impl_,output.data());
        const auto values=output.toVectorAsync(stream);stream.synchronize();double error=0;int observed=0;
        for(int i=0;i<int(near.size());++i)if(values[i].valid)
        {++observed;const auto p=near[i];error=std::max(error,std::abs(values[i].distance-(std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z)-.5)));}
        if(observed!=int(near.size()) || error>3*voxel)throw std::runtime_error("generated ESDF failed resolution-aware sphere check");
        // Finite differences check the derivative of the queried interpolant.
        const Point center{.113,.133,.173};std::vector<Point> fd={center};constexpr double eps=1e-6;
        for(int axis=0;axis<3;++axis)
        {
            Point plus=center,minus=center;double Point::*member=axis==0?&Point::x:(axis==1?&Point::y:&Point::z);
            plus.*member+=eps;minus.*member-=eps;fd.push_back(plus);fd.push_back(minus);
        }
        device_vector<Point> fdp;fdp.copyFromAsync(fd,stream);device_vector<Sample> fdo(7);stream.synchronize();
        query<true,true><<<1,32>>>(7,fdp.data(),nullptr,hash,ev.getHash().impl_,fdo.data());
        const auto fdh=fdo.toVectorAsync(stream);stream.synchronize();double derivative_error=0;
        for(int axis=0;axis<3;++axis)
        {
            if(!fdh[1+2*axis].valid || !fdh[2+2*axis].valid || !fdh[0].valid)throw std::runtime_error("invalid ESDF FD query");
            const double actual=axis==0?fdh[0].gx:(axis==1?fdh[0].gy:fdh[0].gz);
            derivative_error=std::max(derivative_error,std::abs(actual-(fdh[1+2*axis].distance-fdh[2+2*axis].distance)/(2*eps)));
        }
        if(derivative_error>1e-6)throw std::runtime_error("ESDF interpolation derivative check failed");
        // Change a single observed block and propagate a genuine ESDF update.
        const Index3D changed_index(12,9,9);auto changed_host=host.getBlockAtIndex(changed_index);
        auto changed_device=device.getBlockAtIndex(changed_index);
        const Point changed_point{1.,.04,.04};device_vector<Point> changed_query;
        changed_query.copyFromAsync(std::vector<Point>{changed_point},stream);device_vector<Sample> changed_result(1);stream.synchronize();
        query<true,true><<<1,1>>>(1,changed_query.data(),nullptr,hash,ev.getHash().impl_,changed_result.data());
        auto before_change=changed_result.toVectorAsync(stream);stream.synchronize();
        for(int z=0;z<8;++z)for(int y=0;y<8;++y)for(int x=0;x<8;++x)changed_host->voxels[x][y][z].distance=-.01f;
        check(cudaMemcpy(changed_device.get(),changed_host.get(),sizeof(TsdfBlock),cudaMemcpyHostToDevice));
        esdf_begin=std::chrono::steady_clock::now();integrator.integrateBlocks(device,{changed_index},&esdf);check(cudaDeviceSynchronize());
        const double changed_update_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-esdf_begin).count();
        query<true,true><<<1,1>>>(1,changed_query.data(),nullptr,hash,ev.getHash().impl_,changed_result.data());
        auto after_change=changed_result.toVectorAsync(stream);stream.synchronize();
        if(!before_change[0].valid || !after_change[0].valid || before_change[0].distance<.4 || after_change[0].distance>.05)
            throw std::runtime_error("changed-block ESDF update did not propagate");
        device_vector<float> snapshot(dim*dim*dim);device_vector<int> unknown(1);
        __half* destination=nullptr;
#ifdef HORIZON_CUROBO_SCENE
        curobo::robot::SceneConfig cfg;cfg.max_voxel_grids=1;cfg.max_voxels_per_grid=dim*dim*dim;
        curobo::robot::Scene scene(cfg);curobo::robot::VoxelGrid input;
        input.name="nvblox_snapshot";input.dimensions={dim*voxel,dim*voxel,dim*voxel};input.voxel_size=voxel;
        input.esdf.resize(dim*dim*dim,0.f);const auto handle=scene.add_voxel_grid(input);
        destination=reinterpret_cast<__half*>(scene.voxel_grid_cuda_address(handle));
#endif
        auto export_launch=[&]{check(cudaMemset(unknown.data(),0,sizeof(int)));exportDense<<<(dim*dim*dim+255)/256,256>>>(ev.getHash().impl_,snapshot.data(),destination,unknown.data());};
        const auto snapshot_time=timing(export_launch);int unknown_count=0;
        check(cudaMemcpy(&unknown_count,unknown.data(),sizeof(int),cudaMemcpyDeviceToHost));
        if(unknown_count)throw std::runtime_error("unknown voxels in requested ESDF snapshot");
        double half_error=0.;
#ifdef HORIZON_CUROBO_SCENE
        if(reinterpret_cast<__half*>(scene.voxel_grid_cuda_address(handle))!=destination)throw std::runtime_error("cuRobo scene address changed");
        std::vector<__half> halves(dim*dim*dim);check(cudaMemcpy(halves.data(),destination,halves.size()*sizeof(__half),cudaMemcpyDeviceToHost));
        const auto full=snapshot.toVectorAsync(stream);stream.synchronize();
        for(int z=0;z<dim;++z)for(int y=0;y<dim;++y)for(int x=0;x<dim;++x)
            half_error=std::max(half_error,double(std::abs(full[(z*dim+y)*dim+x]-__half2float(halves[(x*dim+y)*dim+z]))));
        if(half_error>.004)throw std::runtime_error("cuRobo FP16 snapshot conversion error too large");
#endif
        std::cout<<"],\"esdf_first_build_ms\":"<<first_esdf_ms<<",\"esdf_full_reupdate_median_ms\":"<<updates[2]
                 <<",\"esdf_observed_test_queries\":"<<observed<<",\"esdf_analytic_max_distance_error\":"<<error
                 <<",\"esdf_gradient_fd_max_error\":"<<derivative_error<<",\"unknown_corner_rejected\":true"
                 <<",\"esdf_changed_block_update_ms\":"<<changed_update_ms<<",\"changed_block_distance_before\":"<<before_change[0].distance
                 <<",\"changed_block_distance_after\":"<<after_change[0].distance<<",\"esdf_full_reupdate_input_changed\":false"
                 <<",\"gpu_dense_snapshot_median_ms\":"<<snapshot_time.first<<",\"gpu_dense_snapshot_p95_ms\":"<<snapshot_time.second
                 <<",\"curobo_scene_gpu_write\":"<<(destination?"true":"false")<<",\"fp16_snapshot_max_error\":"<<half_error<<"}\n";
    }
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
