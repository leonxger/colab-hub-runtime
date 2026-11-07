# runtime

Sources, patches and build scripts of the prebuilt llama.cpp runtimes that the Colab VM boots. Nothing here runs on a user's PC or on the VM: the scripts run in CI inside the pinned CUDA image, and the tarballs they produce are published as release assets.

The patches are derived from llama.cpp (MIT, `patches/*/LICENSE.llama.cpp`) and are byte copies of the lab files, except `patch_loadwait.py` (M15-T12) and `patch_streamprefill.py` (M15-T13), which only this repo has. The upstream `LICENSE` goes into every tarball.

## Layout

| Path | What |
|---|---|
| `versions.env` | every pin: llama.cpp tags and commits, CUDA image and digest, CCCL, owner and name of the release repo |
| `patches/qwen/` | `series` plus two env-gated draft-vocabulary trim diffs for b12015 (MTP and DFlash) and `patch_loadwait.py` (load while downloading, inert unless `LLAMA_LOAD_WAIT` is set) |
| `patches/deepseek/` | `series` plus 24 scripts (`apply_pr30966.py` applies `pr30966.diff`) for b11963, applied in order; the last two are this repo's own: `patch_loadwait.py`, inert unless `LLAMA_LOAD_WAIT` is set, and `patch_streamprefill.py` (the MoE prefill uploads each pinned expert once its chunks are verified, inert unless `LLAMA_MOE_STREAM_PREFILL=1`; `LLAMA_UPLOAD_LOG` gets one `uploaded` line per device copy of a mapped range) |
| `src/pin.c` | `LD_PRELOAD` shim, built into `lib/libpin.so` of the DeepSeek tarball |
| `build/common.sh` | clone at the tag, commit check, apply the series, packing helpers |
| `build/qwen.sh`, `build/deepseek.sh` | one family each: CMake line, targets, tarball layout |
| `build/check_elf.sh` | post-build checks on the staged tree |
| `build/cudart.sh` | optional fallback asset with the CUDA 13 runtime libraries of the image |
| `tests/` | `run.sh` runs every `test_*.sh`; `golden/` holds the expected CMake lines, package lists and the sha256 manifest of the copied files |

## Use

Inside the image named by `CUDA_IMAGE` and `CUDA_IMAGE_DIGEST` with `git python3 cmake ninja-build ccache build-essential libssl-dev` installed:

```
WORK=/work TREE12=<12 hex of the runtime/ tree hash> bash runtime/build/qwen.sh
WORK=/work TREE12=<12 hex> bash runtime/build/deepseek.sh
WORK=/work TREE12=<12 hex> bash runtime/build/cudart.sh
```

`WORK` must be an absolute, empty or missing directory (the scripts refuse a used one). The tarball and its `.sha256` land in `$WORK/out`. Other knobs: `JOBS` (default 4), `CCACHE_MAXSIZE` (default 2G), `SOURCE_DATE_EPOCH` (tar mtime, default 0), `CLONE_TIMEOUT` (seconds, default 600).

| Command | Effect |
|---|---|
| `qwen.sh --print-cmake`, `deepseek.sh --print-cmake` | print the configure line and exit, no environment needed |
| `qwen.sh --list-package`, `deepseek.sh --list-package` | print the tarball name, then the paths it must contain (`bin/test-backend-ops` for Qwen, P18) |
| `qwen.sh --package-only` | pack an existing `$WORK/src` and `$WORK/build` (used by the tests) |
| `common.sh --apply-only qwen DIR` | shallow clone at the pinned tag into `DIR` and apply the series, no build |

A build runs these steps and stops at the first failure: tool and `nvcc` release check (must equal the image's 13.0), shallow clone with commit check, series in order (`git apply` for `.diff`, `python3 -I SCRIPT SRC` for `.py`), configure with the printed line, build of the named targets, staging, `BUILDINFO.json`, `check_elf.sh`, tar.

## Tarballs

| Asset | Contents |
|---|---|
| `llama-qwen-b12015-cu13.0.2-sm80-sm120a-<tree12>.tar.gz` | `bin/llama-server`, `bin/llama-fit-params`, `bin/test-backend-ops`, `bin/*.so` (backends are dynamic, CPU variants included), `py/` (`convert_hf_to_gguf.py`, `conversion/`, `gguf-py/gguf/`), `BUILDINFO.json`, `LICENSE` |
| `llama-deepseek-b11963p24-cu13.0.2-sm120a-<tree12>.tar.gz` | `bin/llama-server`, `bin/llama-fit-params`, `bin/*.so`, `lib/libpin.so`, `BUILDINFO.json`, `LICENSE` |
| `cudart-13.0.2-x64-<tree12>.tar.gz` | `lib/libcudart.so.13*`, `lib/libcublas.so.13*`, `lib/libcublasLt.so.13*`, `BUILDINFO.json`, `LICENSE` (the CUDA EULA when the image has it) |

`<tree12>` is the first 12 hex characters of the `runtime/` tree hash. Binaries carry `RUNPATH $ORIGIN`. `BUILDINFO.json` has no timestamps: tag, commit, the sha256 of every patch, the CUDA image digest, the exact CMake line, the architectures and the targets. Packing is deterministic (sorted names, fixed owner and mtime, `gzip -n`), so equal build outputs give equal bytes.

## CMake lines

The goldens in `tests/golden/*-cmake.txt` are what `--print-cmake` must print.

- Qwen: the upstream release flags at b12015 plus ours, as in the milestone text, with one change: `-DLLAMA_BUILD_TESTS=ON`. At b12015 `tests/` (and so the `test-backend-ops` target) is added only when that option is on (`CMakeLists.txt`: `LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS`); only the named targets are built, so no other test is compiled. `GGML_CUDA_CCCL_VERSION` comes from `CCCL` in `versions.env`.
- DeepSeek: the milestone's flags (explicit AVX-512 ISA set with `GGML_NATIVE=OFF`, `120a-real`, `LLAMA_BUILD_TESTS=OFF`, `LLAMA_OPENSSL=OFF`, `LLAMA_CURL=OFF`, `LLAMA_USE_PREBUILT_UI=OFF`, targets `llama-server llama-fit-params`) plus the parts the tarball needs and that change no code: `-G Ninja`, `RUNPATH $ORIGIN`, `GGML_CUDA_NCCL=OFF`, `LLAMA_BUILD_EXAMPLES=OFF`, ccache launchers. The CPU backend is built in, not dynamic, and the toolkit's bundled CCCL is used (today's lab build), so no `GGML_CUDA_CCCL_VERSION`.

## Checks after the build (`check_elf.sh DIR ARCH...`)

- `libggml-cuda.so` holds exactly the expected cubins (`sm_80` and `sm_120a` for Qwen, `sm_120a` for DeepSeek) according to `cuobjdump --list-elf`.
- No ELF in `bin/` or `lib/` needs a `GLIBC_` symbol above 2.39 (`MAX_GLIBC`).
- `ldd` with `LD_LIBRARY_PATH=bin:lib:/usr/local/cuda/lib64` leaves nothing unresolved except the driver's `libcuda.so.1`, resolves `libcudart`, `libcublas` and `libcublasLt` under `/usr/local/cuda/lib64`, shows no `libcudart.so.12`, and shows `libcudart.so.13` at least once.

## Tests

`bash runtime/tests/run.sh` runs everything offline in under a minute, with no CUDA and no network: the copied files against `golden/patches-sha256.txt` and (when the lab repo is at `$HOME/colab-hub` or `LAB_REPO`) against their sources, the golden CMake lines, `common.sh` against a local fake upstream, packaging of both families from fake build trees, `check_elf.sh` with stub `cuobjdump` and `ldd`, `cudart.sh` from a fake library directory, and syntax.

Set `RT_SRC_QWEN` and `RT_SRC_DEEPSEEK` to patched clones (`common.sh --apply-only ...`) to also check every `-D` option name and target name of the two lines against the real trees, and to build a CPU-only static libllama of each tree and run `tests/loadwait_test.cpp` against it, and for DeepSeek `tests/streamprefill_test.cpp` (a fake GPU device, the prefill under random chunk arrival, then `vm/deepseek/checksizes.py` and `punch.py` on its fixture) (`test_loadwait.sh`: needs `cmake`, `ninja` and `g++`, about 2 min per tree at two jobs on a Raspberry Pi 5; `CMAKE` names another cmake).

Only CI can show: the CUDA build itself, the cubin set and `ldd` of the real binaries, the GLIBC limit, and the `cuobjdump --list-elf` output format that `check_elf.sh` parses.
