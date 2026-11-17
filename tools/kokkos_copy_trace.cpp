// Standalone Kokkos Tools plugin; intentionally outside production targets.
// c++ -shared -fPIC -I /path/to/kokkos/include tools/kokkos_copy_trace.cpp \
//     -o /tmp/horizon_kokkos_copy_trace.so
// KOKKOS_TOOLS_LIBS=/tmp/horizon_kokkos_copy_trace.so benchmarkKyonTrajectory ...
#include <impl/Kokkos_Profiling_C_Interface.h>
#include <cstdio>
#include <inttypes.h>

extern "C" void kokkosp_init_library(int,uint64_t,uint32_t,Kokkos_Profiling_KokkosPDeviceInfo*)
{std::fprintf(stderr,"KOKKOS_TRACE,event,destination,source,bytes,label\n");}

extern "C" void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle destination,const char* destination_label,
    const void*,Kokkos_Profiling_SpaceHandle source,const char* source_label,const void*,uint64_t bytes)
{
    std::fprintf(stderr,"KOKKOS_TRACE,copy,%s,%s,%" PRIu64 ",%s <- %s\n",
                 destination.name,source.name,bytes,destination_label,source_label);
}

extern "C" void kokkosp_allocate_data(Kokkos_Profiling_SpaceHandle space,const char* label,const void*,uint64_t bytes)
{std::fprintf(stderr,"KOKKOS_TRACE,allocate,%s,,%" PRIu64 ",%s\n",space.name,bytes,label);}

extern "C" void kokkosp_deallocate_data(Kokkos_Profiling_SpaceHandle space,const char* label,const void*,uint64_t bytes)
{std::fprintf(stderr,"KOKKOS_TRACE,free,%s,,%" PRIu64 ",%s\n",space.name,bytes,label);}
