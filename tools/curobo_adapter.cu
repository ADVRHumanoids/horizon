// Experimental adapter only. Build with nvcc -O3 -shared -Xcompiler -fPIC
// -arch=sm_86 tools/curobo_adapter.cu -o /tmp/horizon_curobo_adapter.so
#include <cuda_runtime.h>
#include <cstdint>

__device__ void rotation(double w,double x,double y,double z,double* r)
{
    const double inv=rsqrt(w*w+x*x+y*y+z*z);w*=inv;x*=inv;y*=inv;z*=inv;
    r[0]=1-2*(y*y+z*z);r[1]=2*(x*y-z*w);r[2]=2*(x*z+y*w);
    r[3]=2*(x*y+z*w);r[4]=1-2*(x*x+z*z);r[5]=2*(y*z-x*w);
    r[6]=2*(x*z-y*w);r[7]=2*(y*z+x*w);r[8]=1-2*(x*x+y*y);
}
__global__ void adapt(const float* positions,const float* quaternions,const float* jacobians,
    const double* q,const double* root_r,const double* root_t,const int64_t* frames,
    const double* offsets,const int64_t* order,double* centers,double* jac,
    int nodes,int spheres,int frame_count,int dofs)
{
    const int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=nodes*spheres)return;
    const int node=index/spheres,sphere=index%spheres,frame=frames[sphere];
    const float* quat=quaternions+(node*frame_count+frame)*4;
    double link_r[9],local_offset[3],local[3],base_motion[9],base_r[9],base_t[3];
    rotation(quat[0],quat[1],quat[2],quat[3],link_r);
    const double* configuration=q+node*(dofs+7);
    rotation(configuration[6],configuration[3],configuration[4],configuration[5],base_motion);
    for(int a=0;a<3;++a)
    {
        local_offset[a]=0.;base_t[a]=root_t[a];
        for(int b=0;b<3;++b){local_offset[a]+=link_r[3*a+b]*offsets[3*sphere+b];base_t[a]+=root_r[3*a+b]*configuration[b];}
        local[a]=positions[(node*frame_count+frame)*3+a]+local_offset[a];
        for(int b=0;b<3;++b){base_r[3*a+b]=0.;for(int c=0;c<3;++c)base_r[3*a+b]+=root_r[3*a+c]*base_motion[3*c+b];}
    }
    for(int a=0;a<3;++a)
    {
        double p=base_t[a];for(int b=0;b<3;++b)p+=base_r[3*a+b]*local[b];centers[index*3+a]=p;
    }
    const float* jf=jacobians+(node*frame_count+frame)*6*dofs;
    for(int col=0;col<dofs+6;++col)
    {
        double v[3]={0.,0.,0.};
        if(col<3)v[col]=1.;
        else if(col<6)
        {
            const int k=col-3;
            v[(k+1)%3]=-local[(k+2)%3];v[(k+2)%3]=local[(k+1)%3];
        }
        else
        {
            const int k=order[col-6];
            for(int a=0;a<3;++a)
                v[a]=jf[a*dofs+k]+jf[((a+1)%3+3)*dofs+k]*local_offset[(a+2)%3]
                     -jf[((a+2)%3+3)*dofs+k]*local_offset[(a+1)%3];
        }
        for(int a=0;a<3;++a)
        {double d=0.;for(int b=0;b<3;++b)d+=base_r[3*a+b]*v[b];jac[(index*3+a)*(dofs+6)+col]=d;}
    }
}

extern "C" int horizon_curobo_adapt(const float* positions,const float* quaternions,const float* jacobians,
    const double* q,const double* root_r,const double* root_t,const int64_t* frames,
    const double* offsets,const int64_t* order,double* centers,double* jac,
    int nodes,int spheres,int frame_count,int dofs,cudaStream_t stream)
{
    adapt<<<(nodes*spheres+127)/128,128,0,stream>>>(positions,quaternions,jacobians,q,root_r,root_t,
        frames,offsets,order,centers,jac,nodes,spheres,frame_count,dofs);
    return int(cudaPeekAtLastError());
}
