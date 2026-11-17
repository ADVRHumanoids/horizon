# Source this file to select the isolated toolkit for the current shell only.
horizon_cuda13_prefix="${CUDA13_ROOT:-${HOME}/.local/opt/cuda-13.0.2}"
if [ ! -x "${horizon_cuda13_prefix}/bin/nvcc" ]; then
    printf 'CUDA toolkit not found: %s\n' "${horizon_cuda13_prefix}" >&2
    return 1
fi
export CUDA_HOME="${horizon_cuda13_prefix}"
export CUDA_PATH="${horizon_cuda13_prefix}"
export CUDAToolkit_ROOT="${horizon_cuda13_prefix}"
export PATH="${horizon_cuda13_prefix}/bin:${PATH}"
export LD_LIBRARY_PATH="${horizon_cuda13_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset horizon_cuda13_prefix
