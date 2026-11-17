#!/usr/bin/env python3
"""Install isolated NVIDIA CUDA compiler/runtime components, without a driver.

Downloads official redistributable archives and verifies manifest SHA-256s.
Only the requested prefix and cache are written; shell defaults are untouched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

COMPONENTS = ["cuda_nvcc", "cuda_crt", "cuda_cccl", "cuda_cudart", "libnvvm",
              "cuda_nvrtc", "libnvjitlink", "cuda_cupti", "cuda_sanitizer_api",
              "cuda_nvtx", "cuda_profiler_api", "libnpp", "libcurand", "cuda_culibos",
              "libcublas", "libcufft", "libcusolver", "libcusparse", "libnvjpeg",
              "libnvfatbin", "libnvptxcompiler", "cuda_nvdisasm", "cuda_cuobjdump", "cuda_nvml_dev"]
BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="13.0.2")
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--cache", default=Path("/tmp/horizon-cuda-downloads"), type=Path)
    parser.add_argument("--components", nargs="+", default=COMPONENTS,
                        help="Override components, e.g. to extend an existing installation")
    args = parser.parse_args()
    manifest = json.load(urllib.request.urlopen(BASE + f"redistrib_{args.version}.json"))
    args.prefix.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    def fetch(component):
        entry = manifest[component]["linux-x86_64"]
        path = args.cache / Path(entry["relative_path"]).name
        if not path.exists() or checksum(path) != entry["sha256"]:
            print(f"Downloading {component}: {int(entry['size']) / 1e6:.1f} MB", flush=True)
            partial = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(BASE + entry["relative_path"]) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            with partial.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch: {component}")
            partial.rename(path)
        return component, path

    with ThreadPoolExecutor(max_workers=3) as workers:
        archives = list(workers.map(fetch, args.components))
    for component, path in archives:
        print(f"Installing {component}", flush=True)
        with tarfile.open(path) as archive:
            members = []
            for member in archive.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                if ".." in parts or Path(*parts).is_absolute():
                    raise RuntimeError("Unsafe archive member")
                member.name = str(Path(*parts))
                if member.name.upper().startswith(("LICENSE", "EULA")):
                    member.name = str(Path("licenses") / component / member.name)
                members.append(member)
            archive.extractall(args.prefix, members=members, filter="data")
    (args.prefix / "redistrib_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Installed compiler/runtime/development components in {args.prefix}", flush=True)
    print(f"Select with CUDAToolkit_ROOT={args.prefix} and CMAKE_CUDA_COMPILER={args.prefix}/bin/nvcc")


if __name__ == "__main__":
    main()
