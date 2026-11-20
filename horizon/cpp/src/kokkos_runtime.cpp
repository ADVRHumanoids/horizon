#include "horizon/kokkos_runtime.h"
#include <Kokkos_Core.hpp>

namespace horizon::collision
{
struct KokkosRuntime::Impl
{
    Kokkos::ScopeGuard guard;
};
KokkosRuntime::KokkosRuntime() : impl(std::make_unique<Impl>())
{
}
KokkosRuntime::~KokkosRuntime() = default;
} // namespace horizon::collision
