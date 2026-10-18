"""Build faster-whisper.exe (one file) with Nuitka.

Bundles: the CTranslate2 Whisper model in --model_dir, and the two cuBLAS DLLs
that ctranslate2 actually loads on CUDA (cuDNN / nvrtc are never loaded for
Whisper, so they are left out to keep the binary small).

The onefile payload is zstd-compressed and unpacked once into a per-build
cache directory, so later runs start without re-extracting ~2 GB.
"""

import argparse
import os
import subprocess
import sys
import sysconfig

p = argparse.ArgumentParser()
p.add_argument("--model_dir", default="model")
p.add_argument("--output_dir", default="dist")
p.add_argument("--build_id", default="dev", help="cache folder name; change it per release")
p.add_argument("--compiler", default=None, help="e.g. --zig or --msvc=latest")
args = p.parse_args()

if not os.path.isfile(os.path.join(args.model_dir, "model.bin")):
    sys.exit(f"model not found in {args.model_dir}")

cublas_bin = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cublas", "bin")
dlls = ["cublas64_12.dll", "cublasLt64_12.dll"]
for d in dlls:
    if not os.path.isfile(os.path.join(cublas_bin, d)):
        sys.exit(f"missing {d} in {cublas_bin}; pip install nvidia-cublas-cu12")

cmd = [
    sys.executable, "-m", "nuitka",
    "--onefile",
    "--assume-yes-for-downloads",
    f"--output-dir={args.output_dir}",
    "--output-filename=faster-whisper.exe",
    f"--onefile-tempdir-spec={{CACHE_DIR}}/faster-whisper-exe/{args.build_id}",
    "--include-package-data=faster_whisper",
    f"--include-data-dir={args.model_dir}=model",
    *[f"--include-data-files={os.path.join(cublas_bin, d)}=cuda/{d}" for d in dlls],
    "--nofollow-import-to=torch,transformers,tkinter,matplotlib,PIL,scipy,pandas,IPython,pytest,setuptools,hf_xet",
    "whisper_cli.py",
]
if args.compiler:
    cmd.insert(3, args.compiler)

print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
