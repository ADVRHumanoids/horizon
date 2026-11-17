// Optional isolated library comparison; no additional production dependency.
#include <Kokkos_Core.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using Space=Kokkos::DefaultExecutionSpace;
using V3=Kokkos::View<double***,Kokkos::LayoutRight,Space::memory_space>;
using V2=Kokkos::View<double**,Kokkos::LayoutLeft,Space::memory_space>;
using Policy=Kokkos::TeamPolicy<Space>;
using Member=Policy::member_type;

template<class Launch> std::pair<double,double> measure(Launch launch)
{
    for(int i=0;i<10;++i)launch();Space().fence();
    std::vector<double> samples;
    for(int sample=0;sample<20;++sample)
    {
        Kokkos::Timer timer;
        for(int repeat=0;repeat<10;++repeat)launch();
        Space().fence();samples.push_back(timer.seconds()*1e3/10.);
    }
    std::sort(samples.begin(),samples.end());
    return {(.5*(samples[9]+samples[10])),samples[18]};
}

int main(int argc,char** argv)
{
    Kokkos::ScopeGuard runtime(argc,argv);
    std::cout<<std::unitbuf;
    constexpr int nodes=257,nv=30,columns=31,packed=nv*(nv+1)/2,entries=1+nv+packed;
    std::cout<<std::setprecision(8)<<"rows,method,median_ms,p95_ms,max_error\n";
    for(int rows:{94,194,1024})
    {
        V3 j("residual Jacobian and residual",nodes,rows,columns),gram("Gram",nodes,columns,columns);
        V2 result("packed contribution",nodes,entries),expected("reference",nodes,entries);
        Kokkos::parallel_for("initialize",Kokkos::RangePolicy<Space>(0,nodes*rows*columns),KOKKOS_LAMBDA(int i)
        {j(i/(rows*columns),(i/columns)%rows,i%columns)=Kokkos::sin(.01*i);});
        auto custom=[&](){Kokkos::parallel_for("packed reduction",Kokkos::RangePolicy<Space>(0,nodes*entries),KOKKOS_LAMBDA(int i)
        {
            const int node=i/entries,entry=i%entries;int a=nv,b=nv;double scale=1.;
            if(!entry)scale=.5;else if(entry<=nv)a=entry-1;
            else {int k=entry-1-nv;a=0;while(k>=nv-a){k-=nv-a;++a;}b=a+k;}
            double sum=0.;for(int r=0;r<rows;++r)sum+=j(node,r,a)*j(node,r,b);
            result(node,entry)=scale*sum;
        });};
        custom();Kokkos::deep_copy(expected,result);
        auto timing=measure(custom);
        std::cout<<rows<<",custom,"<<timing.first<<','<<timing.second<<",0\n";
        for(int size:{32,64,128})for(int blocked:{0,1})
        {
            auto library=[&]()
            {
                Kokkos::parallel_for("KokkosBatched TeamGemm",Policy(nodes,size),KOKKOS_LAMBDA(const Member& team)
                {
                    auto A=Kokkos::subview(j,team.league_rank(),Kokkos::ALL(),Kokkos::ALL());
                    auto C=Kokkos::subview(gram,team.league_rank(),Kokkos::ALL(),Kokkos::ALL());
                    if(blocked)KokkosBatched::TeamGemm<Member,KokkosBatched::Trans::Transpose,KokkosBatched::Trans::NoTranspose,
                        KokkosBatched::Algo::Gemm::Blocked>::invoke(team,1.,A,A,0.,C);
                    else KokkosBatched::TeamGemm<Member,KokkosBatched::Trans::Transpose,KokkosBatched::Trans::NoTranspose,
                        KokkosBatched::Algo::Gemm::Unblocked>::invoke(team,1.,A,A,0.,C);
                });
                Kokkos::parallel_for("pack Gram",Kokkos::RangePolicy<Space>(0,nodes*entries),KOKKOS_LAMBDA(int i)
                {
                    const int node=i/entries,entry=i%entries;int a=nv,b=nv;double scale=1.;
                    if(!entry)scale=.5;else if(entry<=nv)a=entry-1;
                    else {int k=entry-1-nv;a=0;while(k>=nv-a){k-=nv-a;++a;}b=a+k;}
                    result(node,entry)=scale*gram(node,a,b);
                });
            };
            timing=measure(library);
            const auto actual=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),result);
            const auto reference=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),expected);
            double error=0.;for(int n=0;n<nodes;++n)for(int e=0;e<entries;++e)error=std::max(error,std::abs(actual(n,e)-reference(n,e)));
            if(error>1e-8)return 1;
            std::cout<<rows<<",team"<<size<<(blocked?"_blocked":"_unblocked")<<','<<timing.first<<','<<timing.second<<','<<error<<'\n';
        }
    }
}
