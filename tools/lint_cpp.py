#!/usr/bin/env python3
"""Check all repository C++ files; pass --fix to apply tidy and format fixes."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {".cpp", ".cc", ".cxx", ".c++"}
HEADERS = {".h", ".hh", ".hpp", ".hxx"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build",
                        help="Directory containing compile_commands.json (default: repository build/)")
    parser.add_argument("--fix", action="store_true", help="Apply fixes in place")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--format-only", action="store_true")
    mode.add_argument("--tidy-only", action="store_true")
    parser.add_argument("--clang-format", default="clang-format", metavar="EXECUTABLE")
    parser.add_argument("--clang-tidy", default="clang-tidy", metavar="EXECUTABLE")
    args = parser.parse_args()

    tools = []
    if not args.tidy_only:
        tools.append(args.clang_format)
    if not args.format_only:
        tools.append(args.clang_tidy)
    for tool in tools:
        if shutil.which(tool) is None:
            parser.error(f"Executable not found: {tool}")

    build_dir = args.build_dir.resolve()
    if not args.format_only and not (build_dir / "compile_commands.json").is_file():
        parser.error(f"No compile_commands.json in {build_dir}; configure CMake with "
                     "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON, or use --format-only")

    # Include new files, exclude ignored build/dependency artifacts, and handle
    # filenames containing whitespace without shell expansion.
    result = subprocess.run(["git", "ls-files", "--cached", "--others",
                             "--exclude-standard", "-z"], cwd=ROOT,
                            stdout=subprocess.PIPE, check=True)
    files = sorted({Path(name.decode()) for name in result.stdout.split(b"\0")
                    if name and (ROOT / name.decode()).is_file()
                    and Path(name.decode()).suffix.lower() in SOURCES | HEADERS})
    sources = [path for path in files if path.suffix.lower() in SOURCES]
    if not files:
        print("No C++ files found.")
        return 0

    failed = False

    def run(command):
        nonlocal failed
        completed = subprocess.run(command, cwd=ROOT)
        failed = failed or completed.returncode != 0

    # Tidy may add braces; format afterward so its edits follow .clang-format.
    if not args.format_only:
        print(f"Running clang-tidy on {len(sources)} C++ sources...", flush=True)
        for source in sources:
            command = [args.clang_tidy, "-p", str(build_dir),
                       f"--config-file={ROOT / '.clang-tidy'}",
                       f"--header-filter=^{re.escape(str(ROOT))}/"]
            if args.fix:
                command.append("--fix")
            run(command + [str(ROOT / source)])

    if not args.tidy_only:
        print(f"Running clang-format on {len(files)} C++ sources and headers...", flush=True)
        command = [args.clang_format, f"--style=file:{ROOT / '.clang-format'}"]
        command += ["-i"] if args.fix else ["--dry-run", "--Werror"]
        run(command + [str(ROOT / path) for path in files])

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
