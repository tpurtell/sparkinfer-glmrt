"""CPU and GPU contracts for benchmarks/spark_expert_replay.py."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_head_revision() -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        text=True, capture_output=True,
    )
    if proc.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", proc.stdout.strip()):
        return proc.stdout.strip()
    # Inside an isolated container mount where a git worktree's parent .git dir
    # is unmounted, derive the 40-hex revision from the real source tree bytes.
    h = hashlib.sha256()
    for rel in ("b12x/moe/_shared/trellis_codebooks.py", "benchmarks/spark_expert_replay.py"):
        h.update((REPO_ROOT / rel).read_bytes())
    return h.hexdigest()[:40]


def _load_replay_module():
    bench_dir = str(REPO_ROOT / "benchmarks")
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    path = REPO_ROOT / "benchmarks" / "spark_expert_replay.py"
    spec = importlib.util.spec_from_file_location("spark_expert_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module():
    bench_dir = str(REPO_ROOT / "benchmarks")
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    path = REPO_ROOT / "benchmarks" / "benchmark_mixed_trellis.py"
    spec = importlib.util.spec_from_file_location("benchmark_mixed_trellis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_bucket(
    root: Path,
    rel: str,
    *,
    capacity: int,
    compute: list[int],
    output_dtype: str,
    revision: str,
    codebook: str = "mcg",
    ws: bool = False,
) -> None:
    bucket = root / rel
    bucket.mkdir(parents=True, exist_ok=True)
    obj_sha = hashlib.sha256(f"{revision}:{codebook}:{capacity}".encode()).hexdigest()
    meta = {
        "compute": compute,
        "hidden": 256,
        "intermediate": 128,
        "experts": 4,
        "top_k": 2,
        "bits": [3, 4],
        "capacity": capacity,
        "decode_capacity": capacity <= 80,
        "output_dtype": output_dtype,
        "swiglu_limit": 10.0,
        "direct": False,
        "tile": [64, 128, 64, 128],
        "route_block": 32 if ws else 8,
        "warp_specialized": ws,
        "input_format": "e4m3_k32" if ws else "bf16",
        "sparkinfer_revision": revision,
        "objects": [{"label": "v41_exl3_core", "object_sha256": obj_sha}],
    }
    if codebook != "mcg":
        meta["codebook"] = codebook
    (bucket / "v41_exl3.json").write_text(json.dumps(meta))


def test_replay_buckets_derives_target_and_codebook_from_manifest(tmp_path: Path) -> None:
    replay = _load_replay_module()
    revision = _git_head_revision()
    shape = {"buckets": {"1": "rtx-tp1/m1", "80": "rtx-tp1/m80", "256": "rtx-tp1/m256"}}
    for cap in (1, 80, 256):
        _write_bucket(
            tmp_path / "mcg", f"rtx-tp1/m{cap}", capacity=cap,
            compute=[12, 0], output_dtype="fp32", revision=revision, codebook="mcg", ws=cap >= 256,
        )
        _write_bucket(
            tmp_path / "mul1", f"rtx-tp1/m{cap}", capacity=cap,
            compute=[12, 0], output_dtype="fp32", revision=revision, codebook="mul1", ws=cap >= 256,
        )

    _, mcg_metas, mcg_sel = replay._replay_buckets(shape, tmp_path / "mcg", [1, 8, 256])
    _, mul1_metas, mul1_sel = replay._replay_buckets(shape, tmp_path / "mul1", [1, 8, 256])
    assert mcg_sel == mul1_sel == {1: 1, 8: 80, 256: 256}
    assert mcg_metas[1].get("codebook", "mcg") == "mcg"
    assert mul1_metas[1]["codebook"] == "mul1"

    # Mixed codebooks inside one package root fail closed.
    _write_bucket(
        tmp_path / "mul1", "rtx-tp1/m80", capacity=80,
        compute=[12, 0], output_dtype="fp32", revision=revision, codebook="mcg",
    )
    with pytest.raises(ValueError, match="inconsistent"):
        replay._replay_buckets(shape, tmp_path / "mul1", [1, 8])


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_validate_positive_scale_rejects_nonpositive_or_nonfinite(bad_scale: float) -> None:
    replay = _load_replay_module()
    with pytest.raises(ValueError, match="finite and positive"):
        replay._validate_positive_scale(bad_scale, "global_scale")


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6])
def test_decode_trellis_f16_matches_independent_mcg_and_mul1_on_cpu(bits: int) -> None:
    import numpy as np

    replay = _load_replay_module()
    k_dim, n_dim = 32, 48
    payload = torch.randint(
        -(1 << 31), (1 << 31) - 1, ((k_dim // 16) * (n_dim // 16) * 8 * bits,),
        dtype=torch.int32, generator=torch.Generator().manual_seed(17 + bits),
    )
    mcg = replay._decode_trellis_f16(payload, k_dim, n_dim, bits, "mcg")
    mul1 = replay._decode_trellis_f16(payload, k_dim, n_dim, bits, "mul1")
    assert mcg.dtype == mul1.dtype == torch.float16
    assert not torch.equal(mcg, mul1)

    words = payload.numpy().view(np.uint32).reshape(k_dim // 16, n_dim // 16, 8 * bits)
    kinv = float(np.uint16(0x1EEE).view(np.float16))
    kbias = float(np.uint16(0xC931).view(np.float16))
    for cb, actual in (("mcg", mcg), ("mul1", mul1)):
        expected = np.empty((k_dim // 16, n_dim // 16, 16, 16), dtype=np.float16)
        for lane in range(32):
            for w in range(8):
                end = (lane * 8 + w + 257) * bits
                first, last = (end - 16) // 32, (end - 1) // 32
                merged = (words[..., first % (8 * bits)].astype(np.uint64) << np.uint64(32))
                merged |= words[..., last % (8 * bits)].astype(np.uint64)
                win = (merged >> np.uint64((last + 1) * 32 - end)) & np.uint64(65535)
                if cb == "mcg":
                    val = ((win * np.uint64(0xCBAC1FED)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
                    val = (val & np.uint32(0x8FFF8FFF)) ^ np.uint32(0x3B603B60)
                    dec = ((val & np.uint32(0xFFFF)).astype(np.uint16).view(np.float16)
                           + (val >> np.uint32(16)).astype(np.uint16).view(np.float16)).astype(np.float16)
                else:
                    prod = ((win * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
                    bsum = ((prod & np.uint32(0xFF)) + ((prod >> np.uint32(8)) & np.uint32(0xFF))
                            + ((prod >> np.uint32(16)) & np.uint32(0xFF)) + ((prod >> np.uint32(24)) & np.uint32(0xFF)))
                    h = (np.uint32(0x6400) + bsum).astype(np.uint16).view(np.float16).astype(np.float64)
                    dec = (h * kinv + kbias).astype(np.float16)
                r = (lane % 4) * 2 + (w % 4 // 2) * 8 + w % 2
                c_idx = 2 * (lane // 8 + (4 if w >= 4 else 0)) + ((lane >> 2) & 1)
                expected[..., r, c_idx] = dec
        ref_t = torch.from_numpy(expected.transpose(0, 2, 1, 3).reshape(k_dim, n_dim).copy())
        assert torch.equal(actual.view(torch.uint16), ref_t.view(torch.uint16))


def _build_real_native_wire_so(build_dir: Path, compute: tuple[int, int]) -> Path:
    """Compile a minimal CUDA wire-decode library matching cuteafd_exl3_wire.cu."""
    build_dir.mkdir(parents=True, exist_ok=True)
    cu = build_dir / "wire.cu"
    so = build_dir / "libcuteafd_native.so"
    cu.write_text(
        """
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <new>

namespace {
uint64_t g_hidden = 0;
struct Context { int device; };

__global__ void decode_wire_kernel(const uint8_t* input, __nv_bfloat16* output, uint64_t hidden, uint64_t elements) {
    const uint64_t row_bytes = hidden + hidden / 32;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t row = i / hidden, col = i % hidden;
        __nv_fp8_e4m3 v; v.__x = input[row * row_bytes + col];
        __nv_fp8_e8m0 s; s.__x = input[row * row_bytes + hidden + col / 32];
        output[i] = __float2bfloat16_rn(__fmul_rn(static_cast<float>(v), static_cast<float>(s)));
    }
}
}  // namespace

extern "C" int32_t cuteafd_set_expert_hidden(uint32_t hidden) {
    if (hidden == 0 || hidden % 32 != 0) return cudaErrorInvalidValue;
    g_hidden = hidden;
    return cudaSuccess;
}
extern "C" int32_t cuteafd_exl3_wire_initialize(void** out) {
    if (!out) return cudaErrorInvalidValue;
    auto* ctx = new(std::nothrow) Context;
    if (!ctx) return cudaErrorMemoryAllocation;
    auto st = cudaGetDevice(&ctx->device);
    if (st != cudaSuccess) { delete ctx; return st; }
    *out = ctx;
    return cudaSuccess;
}
extern "C" void cuteafd_exl3_wire_destroy(void* handle) {
    delete static_cast<Context*>(handle);
}
extern "C" int32_t cuteafd_exl3_wire_decode(
    void* handle, const uint8_t* input, uint64_t input_bytes,
    uint16_t* output, uint64_t output_bytes, uint32_t rows, void* stream
) {
    if (!handle || !input || !output || g_hidden == 0 || rows < 1 || rows > 4096) return cudaErrorInvalidValue;
    const uint64_t in_req = uint64_t(rows) * (g_hidden + g_hidden / 32);
    const uint64_t out_req = uint64_t(rows) * g_hidden * 2;
    if (input_bytes < in_req || output_bytes < out_req) return cudaErrorInvalidValue;
    const uint64_t elements = uint64_t(rows) * g_hidden;
    const unsigned blocks = static_cast<unsigned>((elements + 255) / 256);
    decode_wire_kernel<<<blocks < 1024 ? blocks : 1024, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        input, reinterpret_cast<__nv_bfloat16*>(output), g_hidden, elements);
    return cudaGetLastError();
}
"""
    )
    arch = f"sm_{compute[0]}{compute[1]}"
    subprocess.run(
        ["nvcc", "-O2", "-shared", "-Xcompiler", "-fPIC", f"-arch={arch}", str(cu), "-o", str(so)],
        check=True,
    )
    return so


def _export_real_aot_variant(
    out_dir: Path,
    *,
    capacity: int,
    hidden: int,
    intermediate: int,
    experts: int,
    top_k: int,
    bits: tuple[int, int],
    codebook: str,
    output_dtype: str,
    revision: str,
    aot_runtime_dir: Path,
) -> None:
    """Export and link one real CuTe AOT EXL3 variant + Triton route-pack DSO."""
    from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        EXL3_AOT_MAX_SMS,
        compile_mixed_trellis,
        make_mixed_trellis_buffers,
    )
    from b12x.moe._shared.kernels.w4a16.route_pack import compile_w4a16_route_pack_launches

    out_dir.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    compute = [props.major, props.minor]
    workspace_sms = EXL3_AOT_MAX_SMS[(props.major, props.minor)]
    block_m = 8
    route_slots = route_pack_capacity(capacity * top_k, block_m, experts, topk=top_k)[1]
    route_blocks = (route_slots + block_m - 1) // block_m

    launch = compile_mixed_trellis(
        size_m=capacity,
        hidden_size=hidden,
        intermediate_size=intermediate,
        tier0_num_experts=experts,
        tier1_num_experts=experts,
        tier0_bits=bits[0],
        tier1_bits=bits[1],
        route_num_experts=experts,
        top_k=top_k,
        max_m_blocks=route_blocks,
        sms=workspace_sms,
        workspace_sms=workspace_sms,
        max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=(64, 128, 64, 128),
        trellis_codebook=codebook,
        swiglu_limit=10.0,
        moe_block_size=block_m,
        rotation_input_dtype="bf16",
        full_rotation_output_dtype=output_dtype,
        route_ids_dtype=torch.int32,
        direct_topk_routes=False,
    )

    objects = []
    for label, compiled in (("v41_exl3_core", launch.compiled), ("v41_exl3_sum", launch.topk_sum.compiled)):
        compiled.export_to_c(str(out_dir), label, "cuteafd_" + label)
        header = (out_dir / f"{label}.h").read_text()
        wrapper = re.search(r"static inline int32_t (cute_dsl_\w+_wrapper)\((.*?)\) \{", header, re.S)
        assert wrapper is not None
        objects.append({
            "label": label,
            "wrapper": wrapper[1],
            "parameters": [p.strip() for p in wrapper[2].split(",")],
            "object_sha256": hashlib.sha256((out_dir / f"{label}.o").read_bytes()).hexdigest(),
            "header_sha256": hashlib.sha256(header.encode()).hexdigest(),
        })

    buffers = make_mixed_trellis_buffers(launch, device=torch.device("meta"), sms=workspace_sms)
    layouts = {}
    owners: dict[int, str] = {}
    for field in fields(buffers):
        val = getattr(buffers, field.name)
        owner = owners.setdefault(id(val), field.name)
        layouts[field.name] = {
            "shape": list(val.shape),
            "dtype": str(val.dtype),
            "bytes": val.numel() * val.element_size(),
            "allocation": owner,
            "zero_on_create": field.name == "workspace",
        }
    layouts["packed_route_indices"].update(shape=[route_slots], bytes=route_slots * 4)
    layouts["block_expert_ids"].update(shape=[route_blocks], bytes=route_blocks * 4)

    lut_bytes = launch.trellis_lut.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    (out_dir / "trellis_lut.bin").write_bytes(lut_bytes)

    bridge_lines = [
        "#include <new>",
        "#include <mutex>",
        '#include "v41_exl3_core.h"',
        '#include "v41_exl3_sum.h"',
        "struct Context { int device; int32_t grid_cap; };",
        "struct Modules { std::mutex mutex; unsigned users = 0; cuteafd_v41_exl3_core_Kernel_Module_t core{}; cuteafd_v41_exl3_sum_Kernel_Module_t sum{}; };",
        "static Modules modules;",
        "static void unload_modules() { if (modules.sum.module) cudaLibraryUnload(modules.sum.module); if (modules.core.module) cudaLibraryUnload(modules.core.module); modules.sum.module = nullptr; modules.core.module = nullptr; }",
    ]
    for entry in objects:
        label = entry["label"]
        role = label.rsplit("_", 1)[1]
        bridge_lines += [
            f"static int load_{role}(int device) {{",
            f"cudaLibrary_t* library = &modules.{role}.module; cudaError_t status = cudaSuccess;",
            "if (!*library) {",
            "struct { cudaLibrary_t** library; cudaError_t* status; } init{&library, &status};",
            f"_mlir_cuteafd_{label}_cuda_init(reinterpret_cast<void**>(&init));",
            "if (status != cudaSuccess) return int(status); }",
            "struct { cudaLibrary_t** library; int32_t* device; cudaError_t* status; } load{&library, &device, &status};",
            f"_mlir_cuteafd_{label}_cuda_load_to_device(reinterpret_cast<void**>(&load));",
            "return int(status); }",
        ]
    bridge_lines += [
        'extern "C" int cuteafd_exl3_create(void** out) {',
        "if (!out) return int(cudaErrorInvalidValue); *out = nullptr;",
        "Context* ctx = new(std::nothrow) Context; if (!ctx) return int(cudaErrorMemoryAllocation);",
        "cudaError_t status = cudaGetDevice(&ctx->device); int sms = 0;",
        "if (status == cudaSuccess) status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, ctx->device);",
        "if (status != cudaSuccess) { delete ctx; return int(status); }",
        f"ctx->grid_cap = sms * {launch.blocks_per_sm};",
        "std::lock_guard<std::mutex> lock(modules.mutex);",
        "int error = load_core(ctx->device); if (!error) error = load_sum(ctx->device);",
        "if (error) { if (!modules.users) unload_modules(); delete ctx; return error; }",
        "++modules.users; *out = ctx; return 0; }",
        'extern "C" void cuteafd_exl3_destroy(void* opaque) { auto* ctx = static_cast<Context*>(opaque); if (!ctx) return; std::lock_guard<std::mutex> lock(modules.mutex); if (--modules.users == 0) unload_modules(); delete ctx; }',
    ]
    for entry in objects:
        role = entry["label"].rsplit("_", 1)[1]
        call_args = [f"&modules.{role}"]
        decls = []
        ptrs = []
        scalars = []
        for param in entry["parameters"][1:]:
            m = re.fullmatch(r"(.+?)(\w+)", param)
            assert m is not None
            tname, pname = m[1].strip(), m[2]
            if tname == "cudaStream_t":
                call_args.append("static_cast<cudaStream_t>(stream)")
            elif tname == "int32_t":
                idx = len(scalars)
                scalars.append(pname)
                call_args.append(
                    f"(s[{idx}] < ctx->grid_cap ? s[{idx}] : ctx->grid_cap)"
                    if pname == "grid_x"
                    else f"s[{idx}]"
                )
            elif tname == "void *":
                idx = len(ptrs)
                ptrs.append(pname)
                call_args.append(f"p[{idx}]")
            else:
                idx = len(ptrs)
                ptrs.append(pname)
                ttype = tname[:-1].strip()
                decls.append(f"{ttype} {pname}{{p[{idx}]}};")
                call_args.append(f"&{pname}")
        entry["pointer_slots"] = ptrs
        entry["scalar_slots"] = scalars
        bridge_lines += [
            f'extern "C" int cuteafd_exl3_{role}(void* opaque, void* const* p, const int32_t* s, void* stream) {{',
            "auto* ctx = static_cast<Context*>(opaque);",
            *decls,
            f'return {entry["wrapper"]}({", ".join(call_args)});',
            "}",
        ]
    bridge_cc = out_dir / "v41_exl3_bridge.cc"
    bridge_cc.write_text("\n".join(bridge_lines) + "\n")
    subprocess.run(
        [
            "g++", "-O2", "-shared", "-fPIC", "-std=c++17",
            "-I/usr/local/cuda/include",
            str(bridge_cc), str(out_dir / "v41_exl3_core.o"), str(out_dir / "v41_exl3_sum.o"),
            "-L/usr/local/cuda/lib64", "-lcudart",
            f"-L{aot_runtime_dir}", "-lcute_dsl_runtime", f"-Wl,-rpath,{aot_runtime_dir}",
            "-Wl,-z,defs", "-o", str(out_dir / "libcuteafd_exl3.so"),
        ],
        check=True,
    )

    routes_dir = out_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    plan = compile_w4a16_route_pack_launches(
        tokens=capacity, topk=top_k, block_size=block_m, num_experts=experts, ordinal=0
    )
    stages = (
        [("small_prefix", [0, 1, 2, 3, 4, 5, 6], True, "1")]
        if plan.use_small_prefix
        else [
            ("count", [0, 1, 6], True, "(live + 1023) / 1024"),
            ("prefix", [6, 4, 5], False, "1"),
            ("post_prefix", [2, 3, 5], True, str((max(plan.max_packed_routes, plan.max_route_blocks) + 255) // 256)),
        ]
    ) + [("sort", [0, 1, 2, 5], True, "(live + 255) / 256")]
    r_src = ["#include <cuda.h>", "#include <cstdint>", "#include <new>"]
    r_objs = []
    for name, indices, has_live, grid in stages:
        mapped = name not in ("prefix", "post_prefix")
        rk = plan.programs[name, torch.int32, mapped]
        cubin = rk.asm["cubin"]
        r_src += [
            f"alignas(16) static const unsigned char code_{name}[] = {{",
            ",".join(str(b) for b in cubin),
            "};",
        ]
        r_objs.append((name, rk.name, rk.metadata.num_warps * 32, rk.metadata.shared, indices, has_live, grid))
    r_src += [
        f"struct Context {{ CUmodule modules[{len(r_objs)}]{{}}; CUfunction functions[{len(r_objs)}]{{}}; }};",
        'extern "C" int cuteafd_exl3_routes_destroy(void* opaque) { auto* ctx = static_cast<Context*>(opaque); if (!ctx) return CUDA_SUCCESS; for (auto m : ctx->modules) if (m) cuModuleUnload(m); delete ctx; return CUDA_SUCCESS; }',
        'extern "C" int cuteafd_exl3_routes_create(void** output) {',
        "auto* ctx = new(std::nothrow) Context; if (!ctx) return CUDA_ERROR_OUT_OF_MEMORY; CUresult err = CUDA_SUCCESS;",
    ]
    for idx, (stage, sym, _t, shmem, *_rest) in enumerate(r_objs):
        r_src += [
            f"if (!err) err = cuModuleLoadData(&ctx->modules[{idx}], code_{stage});",
            f'if (!err) err = cuModuleGetFunction(&ctx->functions[{idx}], ctx->modules[{idx}], "{sym}");',
        ]
        if shmem > 49152:
            r_src.append(
                f"if (!err) err = cuFuncSetAttribute(ctx->functions[{idx}], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {shmem});"
            )
    r_src += [
        "if (err) { cuteafd_exl3_routes_destroy(ctx); return err; } *output = ctx; return CUDA_SUCCESS; }",
        'extern "C" int cuteafd_exl3_routes_launch(void* opaque, void* const* pointers, const uint64_t*, int32_t rows, void* stream) {',
        "auto* ctx = static_cast<Context*>(opaque); CUdeviceptr p[7];",
        "for (int i = 0; i < 7; ++i) p[i] = reinterpret_cast<CUdeviceptr>(pointers[i]);",
        f"int32_t live = rows * {top_k}; CUdeviceptr null_scratch = 0; auto cs = static_cast<CUstream>(stream); CUresult err = CUDA_SUCCESS;",
    ]
    if not plan.use_small_prefix:
        r_src.append(f"err = cuMemsetD32Async(p[6], 0, {experts}, cs); if (err) return err;")
    for idx, (_stage, _sym, threads, shmem, indices, has_live, grid) in enumerate(r_objs):
        ka = [f"&p[{j}]" for j in indices] + (["&live"] if has_live else []) + ["&null_scratch", "&null_scratch"]
        r_src += [
            "{ void* args[] = {" + ",".join(ka) + "};",
            f"err = cuLaunchKernel(ctx->functions[{idx}], {grid}, 1, 1, {threads}, 1, 1, {shmem}, cs, args, nullptr); if (err) return err; }}",
        ]
    r_src.append("return CUDA_SUCCESS; }")
    routes_cc = routes_dir / "v41_exl3_routes.cc"
    routes_cc.write_text("\n".join(r_src) + "\n")
    subprocess.run(
        [
            "g++", "-O2", "-shared", "-fPIC", "-std=c++17",
            "-I/usr/local/cuda/include", str(routes_cc),
            "-L/usr/local/cuda/lib64", "-L/usr/local/cuda/lib64/stubs", "-lcuda",
            "-Wl,-z,defs", "-o", str(routes_dir / "libv41_exl3_routes.so"),
        ],
        check=True,
    )
    routes_json = routes_dir / "v41_exl3_routes.json"
    routes_json.write_text(json.dumps({
        "schema": "cuteafd.v41-exl3-routes-aot.v1",
        "sparkinfer_revision": revision,
        "compute": compute,
        "capacity": capacity,
        "topk": top_k,
        "experts": experts,
        "block_size": block_m,
    }, indent=2) + "\n")

    manifest = {
        "schema": "cuteafd.v41-exl3-aot.v1",
        "sparkinfer_revision": revision,
        "compute": compute,
        "sms": workspace_sms,
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "top_k": top_k,
        "capacity": capacity,
        "decode_capacity": capacity <= 80,
        "output_dtype": output_dtype,
        "bits": list(bits),
        "swiglu_limit": 10.0,
        "direct": False,
        "route_slots": route_slots,
        "route_blocks": route_blocks,
        "tile": [64, 128, 64, 128],
        "blocks_per_sm": launch.blocks_per_sm,
        "shared_memory_bytes": launch.shared_memory_bytes,
        "buffers": layouts,
        "objects": objects,
        "requires_route_preparation": True,
        "trellis_lut": {
            "file": "trellis_lut.bin",
            "bytes": len(lut_bytes),
            "sha256": hashlib.sha256(lut_bytes).hexdigest(),
        },
        "route_preparation": {
            "manifest": "routes/v41_exl3_routes.json",
            "sha256": hashlib.sha256(routes_json.read_bytes()).hexdigest(),
        },
    }
    if codebook != "mcg":
        manifest["codebook"] = codebook
    (out_dir / "v41_exl3.json").write_text(json.dumps(manifest, indent=2) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for real AOT sealed replay verification")
def test_legacy_unpaired_mcg_replay_bypasses_jit_reference_and_preserves_schema_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real legacy CLI invocation path with real AOT binaries and honest provenance."""
    pytest.importorskip("cutlass")
    monkeypatch.setenv("B12X_COMPILE_DISK_CACHE", "0")
    replay = _load_replay_module()
    bmt = _load_benchmark_module()

    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    if (cap_major, cap_minor) not in ((12, 0), (12, 1)):
        pytest.skip("EXL3 AOT replay requires SM120 or SM121")

    revision = _git_head_revision()
    aot_runtime_dir = Path(
        subprocess.check_output(
            [sys.executable, "-m", "cutlass.cute.export.aot_config", "--libdir"], text=True
        ).strip()
    )
    native_so = _build_real_native_wire_so(tmp_path / "native", (cap_major, cap_minor))
    output_dtype = "bf16" if (cap_major, cap_minor) == (12, 1) else "fp32"
    hidden, inter, experts, top_k, bits = 256, 256, 4, 2, (3, 4)

    for codebook in ("mcg", "mul1"):
        for capacity in (1, 80):
            _export_real_aot_variant(
                tmp_path / codebook / f"role/m{capacity}",
                capacity=capacity,
                hidden=hidden,
                intermediate=inter,
                experts=experts,
                top_k=top_k,
                bits=bits,
                codebook=codebook,
                output_dtype=output_dtype,
                revision=revision,
                aot_runtime_dir=aot_runtime_dir,
            )

    shape = {
        "id": "tiny-exl3",
        "format": "exl3",
        "hidden": hidden,
        "intermediate": inter,
        "experts": experts,
        "top_k": top_k,
        "bits": list(bits),
        "projection_bits": [[3, 4, 3, 4], [4, 3, 4, 3], [3, 4, 3, 4]],
        "compact": False,
        "expert_widths": [inter] * experts,
        "epilogue": "sum",
        "buckets": {"1": "role/m1", "80": "role/m80"},
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps({"schema": "spark-expert-replay.v1", "rows": [1, 8], "shapes": [shape]}))

    # 1. Real legacy unpaired MCG CLI invocation MUST bypass _own_codebook_reference completely.
    real_own_ref = replay._own_codebook_reference
    replay._REF_GEMM_CACHE.clear()
    replay._REF_HAD_CACHE.clear()
    replay._REF_ACT_CACHE.clear()
    replay._REF_FC2_ROT_CACHE.clear()

    def forbid_jit_ref(*_a, **_kw):
        raise AssertionError("legacy unpaired MCG replay must not launch independent JIT reference")

    monkeypatch.setattr(replay, "_own_codebook_reference", forbid_jit_ref)
    legacy_out = tmp_path / "legacy-mcg.json"
    legacy_args = bmt.benchmark_parser().parse_args([
        "--spark-replay", str(fixture_path),
        "--aot-root", str(tmp_path / "mcg"),
        "--replay-native-library", str(native_so),
        "--replay-shape", "tiny-exl3",
        "--rows", "1,8",
        "--warmup", "1",
        "--rounds", "1",
        "--replays", "2",
        "--output", str(legacy_out),
    ])
    replay.replay_spark_shapes(legacy_args, bmt._AotReplayVariant, bmt.PREFILL_SOURCE_SHA256)
    assert not replay._REF_GEMM_CACHE
    assert not replay._REF_HAD_CACHE

    legacy_json = json.loads(legacy_out.read_text())
    assert set(legacy_json.keys()) == {
        "id", "source_sha256", "device", "prefix_exact", "routes_sha256",
        "cells", "libraries", "fixture_sha256",
    }
    assert legacy_json["source_sha256"] == bmt.PREFILL_SOURCE_SHA256
    assert legacy_json["fixture_sha256"] == hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    assert len(legacy_json["libraries"]) == 5  # native + m1(core,routes) + m80(core,routes)
    for lib_entry in legacy_json["libraries"]:
        lib_file = Path(lib_entry["path"])
        assert lib_file.is_file()
        assert lib_entry["sha256"] == hashlib.sha256(lib_file.read_bytes()).hexdigest()
    for cell in legacy_json["cells"]:
        assert set(cell.keys()) == {
            "rows", "capacity", "distinct", "weight_bytes", "digest", "manifest_sha256",
            "core_sha256", "tile", "route_block", "ws", "median_us", "samples", "min_us", "max_us",
        }
        manifest_file = tmp_path / "mcg" / f"role/m{cell['capacity']}" / "v41_exl3.json"
        manifest_meta = json.loads(manifest_file.read_text())
        assert manifest_meta["sparkinfer_revision"] == revision
        assert cell["manifest_sha256"] == hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        assert cell["core_sha256"] == manifest_meta["objects"][0]["object_sha256"]

    # 2. Explicit opt-in --verify-own-codebook runs _own_codebook_reference and preserves exact schema + numeric digest.
    ref_calls: list[tuple[str, int]] = []

    def spy_own_ref(s, w, x, ids, gates, meta, rows):
        ref_calls.append((meta.get("codebook", "mcg"), rows))
        return real_own_ref(s, w, x, ids, gates, meta, rows)

    monkeypatch.setattr(replay, "_own_codebook_reference", spy_own_ref)
    verify_out = tmp_path / "verify-mcg.json"
    verify_args = bmt.benchmark_parser().parse_args([
        "--spark-replay", str(fixture_path),
        "--aot-root", str(tmp_path / "mcg"),
        "--replay-native-library", str(native_so),
        "--replay-shape", "tiny-exl3",
        "--rows", "1,8",
        "--verify-own-codebook",
        "--correctness-only",
        "--output", str(verify_out),
    ])
    replay.replay_spark_shapes(verify_args, bmt._AotReplayVariant, bmt.PREFILL_SOURCE_SHA256)
    assert ref_calls == [("mcg", 1), ("mcg", 8)]
    verify_json = json.loads(verify_out.read_text())
    assert set(verify_json.keys()) == set(legacy_json.keys())
    assert [c["digest"] for c in verify_json["cells"]] == [c["digest"] for c in legacy_json["cells"]]

    # 3. Unpaired MUL1 replay automatically runs own-codebook bitwise verification.
    ref_calls.clear()
    mul1_out = tmp_path / "unpaired-mul1.json"
    mul1_args = bmt.benchmark_parser().parse_args([
        "--spark-replay", str(fixture_path),
        "--aot-root", str(tmp_path / "mul1"),
        "--replay-native-library", str(native_so),
        "--replay-shape", "tiny-exl3",
        "--rows", "1,8",
        "--correctness-only",
        "--output", str(mul1_out),
    ])
    replay.replay_spark_shapes(mul1_args, bmt._AotReplayVariant, bmt.PREFILL_SOURCE_SHA256)
    assert ref_calls == [("mul1", 1), ("mul1", 8)]
    mul1_json = json.loads(mul1_out.read_text())
    assert mul1_json["codebook"] == "mul1"
    assert all(c["own_ref_bitwise"] is True for c in mul1_json["cells"])

    # 4. Paired MCG vs MUL1 replay checks both arms bitwise, retains MCG digests, and fails closed on mismatch.
    ref_calls.clear()
    paired_out = tmp_path / "paired.json"
    paired_args = bmt.benchmark_parser().parse_args([
        "--spark-replay", str(fixture_path),
        "--aot-root", str(tmp_path / "mcg"),
        "--compare-aot-root", str(tmp_path / "mul1"),
        "--replay-native-library", str(native_so),
        "--replay-shape", "tiny-exl3",
        "--rows", "1,8",
        "--correctness-only",
        "--output", str(paired_out),
    ])
    replay.replay_spark_shapes(paired_args, bmt._AotReplayVariant, bmt.PREFILL_SOURCE_SHA256)
    assert ref_calls == [("mcg", 1), ("mul1", 1), ("mcg", 8), ("mul1", 8)]
    paired_json = json.loads(paired_out.read_text())
    assert [c["digest"] for c in paired_json["cells"]] == [c["digest"] for c in legacy_json["cells"]]
    assert [c["compare"]["digest"] for c in paired_json["cells"]] == [c["digest"] for c in mul1_json["cells"]]
    assert all(c["own_ref_bitwise"] is True and c["compare"]["own_ref_bitwise"] is True for c in paired_json["cells"])

    def bad_mul1_ref(s, w, x, ids, gates, meta, rows):
        out = real_own_ref(s, w, x, ids, gates, meta, rows).clone()
        if meta.get("codebook") == "mul1":
            out.reshape(-1)[0] = out.reshape(-1)[0] + 1.0
        return out

    monkeypatch.setattr(replay, "_own_codebook_reference", bad_mul1_ref)
    with pytest.raises(ValueError, match="compare own-codebook bitwise reference mismatch"):
        replay.replay_spark_shapes(paired_args, bmt._AotReplayVariant, bmt.PREFILL_SOURCE_SHA256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for kernel vs own-codebook bitwise check")
@pytest.mark.parametrize("codebook", ["mcg", "mul1"])
@pytest.mark.parametrize("activation,swiglu_limit", [("silu", 10.0), ("situ", None), ("silu_post_clamp", 7.0)])
@pytest.mark.parametrize("compact", [False, True])
def test_own_codebook_reference_matches_compiled_mixed_trellis_bitwise_on_gpu(
    codebook: str, activation: str, swiglu_limit: float | None, compact: bool,
) -> None:
    pytest.importorskip("cutlass")
    from b12x.moe._shared.kernels.w4a16 import mixed_trellis as mix
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_trellis256_moe_weights
    from dataclasses import replace

    replay = _load_replay_module()
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    hidden, inter, experts, top_k, bits = 256, 256, 4, 2, [3, 4]
    widths = [256, 128, 256, 128] if compact else [256] * 4
    shape = {
        "id": "gpu-ref", "format": "exl3", "hidden": hidden, "intermediate": inter,
        "experts": experts, "top_k": top_k, "bits": bits,
        "projection_bits": [[3, 4, 3, 4], [4, 3, 4, 3], [3, 4, 3, 4]],
        "compact": compact, "expert_widths": widths, "epilogue": "sum",
        "global_scale": 1.0, "svh_scale": 0.01,
    }
    ptr, scalar = replay._weights(shape, seed=123)
    output_dtype = "bf16" if (props.major, props.minor) == (12, 1) else "fp32"
    launch = mix.compile_mixed_trellis(
        size_m=8, hidden_size=hidden, intermediate_size=inter,
        tier0_num_experts=experts, tier1_num_experts=experts,
        tier0_bits=bits[0], tier1_bits=bits[1],
        route_num_experts=experts, top_k=top_k, max_m_blocks=16,
        sms=props.multi_processor_count, max_shared_mem=props.shared_memory_per_block_optin,
        force_tile_config=(64, 128, 64, 128), trellis_codebook=codebook,
        activation=activation, swiglu_limit=swiglu_limit, moe_block_size=8,
        rotation_input_dtype="bf16", full_rotation_output_dtype=output_dtype,
        direct_topk_routes=False, compact_weights=compact,
    )
    gate_counts = (scalar["tier0_gate_experts"], scalar["tier1_gate_experts"])
    up_counts = (scalar["tier0_up_experts"], scalar["tier1_up_experts"])
    fc2_counts = (scalar["tier0_fc2_experts"], scalar["tier1_fc2_experts"])
    tiers = []
    for tier_idx, bit in enumerate(bits):
        tmpl = prepare_trellis256_moe_weights(
            hidden_size=hidden, intermediate_size=inter, num_experts=1,
            activation=activation, fc1_tile_n=128, fc2_tile_n=128, device=device,
            seed=1, params_dtype=torch.float16, w13_layout="trellis_t256_proj",
            trellis_bits=bit, codebook=codebook, tile_config=(64, 128, 64, 128),
        )
        tiers.append(replace(
            tmpl, num_experts=experts,
            w13=ptr[f"t{tier_idx}_w13_ptr"], w2=ptr[f"t{tier_idx}_w2_ptr"],
            w13_global_scale=ptr[f"t{tier_idx}_w13_global_ptr"],
            w2_global_scale=ptr[f"t{tier_idx}_w2_global_ptr"][:fc2_counts[tier_idx]],
        ))
    rotations = mix.MixedTrellisRotations(
        ptr["intermediate_rotations_ptr"].reshape(-1),
        ptr["gate_suh_ptr"].reshape(-1),
        ptr["up_suh_ptr"].reshape(-1),
        ptr["svh_ptr"].reshape(-1),
    )
    descriptor = ptr["descriptor_map_ptr"].reshape(-1)
    descriptor._mt_projection_counts = (gate_counts, up_counts)
    binding = mix.bind_mixed_trellis(
        tiers[0], tiers[1], ptr["global_to_combined_ptr"], descriptor, rotations, launch,
        gate_experts=gate_counts, up_experts=up_counts,
    )
    buffers = mix.make_mixed_trellis_buffers(launch, device=device, sms=props.multi_processor_count)

    rng = torch.Generator(device=device).manual_seed(99)
    wire_x = torch.cat((
        (torch.randn((5, hidden), device=device, generator=rng) * 2.0).to(torch.float8_e4m3fn).view(torch.uint8),
        torch.full((5, hidden // 32), 121, device=device, dtype=torch.uint8),
    ), dim=1)
    exp = wire_x[:, hidden:].float() - 127.0
    x_bf16 = (wire_x[:, :hidden].contiguous().view(torch.float8_e4m3fn).float().reshape(5, hidden // 32, 32)
              * torch.exp2(exp).unsqueeze(-1)).reshape(5, hidden).bfloat16()
    ids = torch.tensor([[0, 1], [2, 3], [1, 2], [3, 0], [0, 3]], device=device, dtype=torch.int32)
    gates = torch.randn((5, top_k), device=device, generator=rng).softmax(-1)

    actual = mix.run_bound_mixed_trellis(x_bf16, gates, ids, binding, buffers).clone()
    meta = {
        "codebook": codebook, "tile": [64, 128, 64, 128],
        "activation": activation, "swiglu_limit": swiglu_limit, "output_dtype": output_dtype,
    }
    expected = replay._own_codebook_reference(shape, (ptr, scalar), wire_x, ids, gates, meta, 5)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "raw_uuid,expected_nvml_uuid",
    [
        ("12345678-9abc-def0-1234-56789abcdef0", "GPU-12345678-9abc-def0-1234-56789abcdef0"),
        ("GPU-fedcba98-7654-3210-fedc-ba9876543210", "GPU-fedcba98-7654-3210-fedc-ba9876543210"),
    ],
)
def test_sample_nvml_state_resolves_by_cuda_uuid_under_remapped_visible_device(
    monkeypatch: pytest.MonkeyPatch,
    raw_uuid: str,
    expected_nvml_uuid: str,
) -> None:
    import types
    from types import SimpleNamespace

    replay = _load_replay_module()
    bench = _load_benchmark_module()

    # Simulate CUDA_VISIBLE_DEVICES=1: visible CUDA ordinal 0 maps to physical GPU 1's UUID.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(replay.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        replay.torch.cuda,
        "get_device_properties",
        lambda dev: SimpleNamespace(uuid=raw_uuid) if dev == 0 else (_ for _ in ()).throw(AssertionError(dev)),
    )

    seen_uuids: list[str] = []
    fake_handle = object()
    fake_pynvml = types.ModuleType("pynvml")
    fake_pynvml.NVML_CLOCK_SM = 1
    fake_pynvml.NVML_CLOCK_MEM = 2
    fake_pynvml.NVML_TEMPERATURE_GPU = 0
    fake_pynvml.nvmlInit = lambda: None

    def _forbid_by_index(_idx):
        raise AssertionError("NVML must resolve by CUDA device UUID, never by CUDA ordinal")

    def _by_uuid(uuid: str):
        seen_uuids.append(uuid)
        assert uuid == expected_nvml_uuid
        return fake_handle

    fake_pynvml.nvmlDeviceGetHandleByIndex = _forbid_by_index
    fake_pynvml.nvmlDeviceGetHandleByUUID = _by_uuid
    fake_pynvml.nvmlDeviceGetClockInfo = lambda h, kind: 2520 if (h is fake_handle and kind == 1) else 14001
    fake_pynvml.nvmlDeviceGetPowerUsage = lambda h: 412340 if h is fake_handle else 0
    fake_pynvml.nvmlDeviceGetEnforcedPowerLimit = lambda h: 600000 if h is fake_handle else 0
    fake_pynvml.nvmlDeviceGetTemperature = lambda h, _t: 58 if h is fake_handle else 0
    fake_pynvml.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 0x4 if h is fake_handle else 0
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    state = replay._sample_nvml_state()
    assert seen_uuids == [expected_nvml_uuid]
    assert state == {
        "sm_clock_mhz": 2520,
        "mem_clock_mhz": 14001,
        "power_w": 412.34,
        "power_limit_w": 600.0,
        "temp_c": 58,
        "throttle_reasons": "0x4",
    }
    assert bench._power_limit_w(SimpleNamespace(uuid=raw_uuid)) == 600.0
    assert seen_uuids == [expected_nvml_uuid, expected_nvml_uuid]


def test_paired_replay_compares_all_executable_recipe_discriminators_and_allows_codebook_pair(
    tmp_path: Path,
) -> None:
    replay = _load_replay_module()
    rev_mcg = "1" * 40
    rev_mul1 = "2" * 40
    shape = {"buckets": {"1": "p/m1", "80": "p/m80", "256": "p/m256"}}

    def write_pair(*, compare_overrides_by_cap: dict[int, dict[str, object]] | None = None):
        mcg_root = tmp_path / "mcg"
        mul1_root = tmp_path / "mul1"
        for cap in (1, 80, 256):
            ws = cap >= 256
            base_meta = {
                "schema": "cuteafd.v41-exl3-aot.v1",
                "sparkinfer_revision": rev_mcg,
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                "compute": [12, 0],
                "sms": 188,
                "physical_sms": 188,
                "hidden": 4096,
                "intermediate": 2048,
                "experts": 288,
                "top_k": 8,
                "capacity": cap,
                "decode_capacity": cap <= 80,
                "output_dtype": "fp32",
                "bits": [3, 4],
                "swiglu_limit": 10.0,
                "direct": False,
                "route_slots": cap * 8 + 64,
                "route_blocks": (cap * 8 + 64) // (32 if ws else 8),
                "tile": [64, 256, 64, 256],
                "route_block": 32 if ws else 8,
                "warp_specialized": ws,
                "ws_input_stages": 3 if ws else None,
                "ws_dynamic_tiles": ws,
                "input_format": "e4m3_k32" if ws else "bf16",
                "blocks_per_sm": 1,
                "shared_memory_bytes": 98304,
                "buffers": {"output": {"shape": [cap, 4096], "dtype": "torch.float32", "bytes": cap * 16384}},
                "unique_execution_buffer_bytes": cap * 16384,
                "requires_route_preparation": True,
                "required_link_libraries": ["cudart", "cute_dsl_runtime"],
                "objects": [
                    {
                        "label": "v41_exl3_core",
                        "wrapper": "cute_dsl_cuteafd_v41_exl3_core_wrapper",
                        "parameters": ["void *module", "void *p0", "int32_t s0", "cudaStream_t stream"],
                        "pointer_slots": ["p0"],
                        "scalar_slots": ["s0"],
                        "object_sha256": "a" * 64,
                        "header_sha256": "b" * 64,
                    }
                ],
                "trellis_lut": {"file": "trellis_lut.bin", "bytes": 2048, "sha256": "c" * 64},
                "route_preparation": {"manifest": "routes/v41_exl3_routes.json", "sha256": "d" * 64},
                "native_execution_verified": False,
            }
            cand_meta = {
                **base_meta,
                "codebook": "mul1",
                "sparkinfer_revision": rev_mul1,
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "physical_sms": 188,
                "native_execution_verified": True,
                "objects": [
                    {
                        **base_meta["objects"][0],
                        "object_sha256": "e" * 64,
                        "header_sha256": "f" * 64,
                    }
                ],
                "trellis_lut": {"file": "trellis_lut.bin", "bytes": 2048, "sha256": "9" * 64},
                "route_preparation": {"manifest": "routes/v41_exl3_routes.json", "sha256": "8" * 64},
            }
            if compare_overrides_by_cap and cap in compare_overrides_by_cap:
                cand_meta.update(compare_overrides_by_cap[cap])
            for root, m in ((mcg_root, base_meta), (mul1_root, cand_meta)):
                bdir = root / f"p/m{cap}"
                bdir.mkdir(parents=True, exist_ok=True)
                (bdir / "v41_exl3.json").write_text(json.dumps(m))
        return mcg_root, mul1_root

    # 1. Valid codebook + provenance-only pair passes across all buckets.
    mcg_root, mul1_root = write_pair()
    _, metas, selected = replay._replay_buckets(shape, mcg_root, [1, 8, 256])
    _, c_metas, c_selected = replay._replay_buckets(shape, mul1_root, [1, 8, 256])
    replay._validate_paired_replay_buckets(metas, selected, c_metas, c_selected)

    # 2. Previously unchecked disparate launch/recipe discriminators fail closed.
    disparate_prefill_cases = [
        ({"ws_dynamic_tiles": False}, "ws_dynamic_tiles"),
        ({"ws_input_stages": 4}, "ws_input_stages"),
        ({"swiglu_limit": None}, "swiglu_limit"),
        ({"activation": "situ"}, "activation"),
        ({"token_major_rotation": True}, "token_major_rotation"),
        ({"fused_input_rotation": True}, "fused_input_rotation"),
        ({"blocks_per_sm": 2}, "blocks_per_sm"),
        ({"shared_memory_bytes": 65536}, "shared_memory_bytes"),
        ({"descriptor_rows": 7}, "descriptor_rows"),
        ({"sms": 48}, "sms"),
        ({"unique_execution_buffer_bytes": 999999}, "unique_execution_buffer_bytes"),
    ]
    for override, expected_field in disparate_prefill_cases:
        mcg_root, mul1_root = write_pair(compare_overrides_by_cap={256: override})
        _, metas, selected = replay._replay_buckets(shape, mcg_root, [1, 8, 256])
        _, c_metas, c_selected = replay._replay_buckets(shape, mul1_root, [1, 8, 256])
        with pytest.raises(ValueError, match=f"paired EXL3 recipe mismatch on .*{expected_field}.* at cap 256"):
            replay._validate_paired_replay_buckets(metas, selected, c_metas, c_selected)

    # 3. Disparate decode_schedule on m1+m80 prefix buckets fails closed even when rows=[256] selects only cap 256.
    mcg_root, mul1_root = write_pair(
        compare_overrides_by_cap={1: {"decode_schedule": "gb10"}, 80: {"decode_schedule": "gb10"}}
    )
    _, metas, selected = replay._replay_buckets(shape, mcg_root, [256])
    _, c_metas, c_selected = replay._replay_buckets(shape, mul1_root, [256])
    with pytest.raises(ValueError, match="paired EXL3 recipe mismatch on decode_schedule at cap 1"):
        replay._validate_paired_replay_buckets(metas, selected, c_metas, c_selected)

