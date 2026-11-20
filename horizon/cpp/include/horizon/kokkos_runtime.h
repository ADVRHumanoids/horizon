#pragma once
#include <memory>

namespace horizon::collision
{
// Own the Kokkos runtime. Solvers and costs must be destroyed before this
// object.
class KokkosRuntime
{
  public:
    KokkosRuntime();
    ~KokkosRuntime();
    KokkosRuntime(const KokkosRuntime &) = delete;
    KokkosRuntime &operator=(const KokkosRuntime &) = delete;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace horizon::collision
