# Disk embedding transports

Status: implemented. Disk transport selection is internal to the shared row
cache used by Engram and PLE embedding. Set `B12X_DISK_BACKEND=gds` before
constructing a disk table to select cuFile with GDS and CPU compatibility fallback. Unset or `io_uring`
selects io_uring. Other values fail. Each cache retains its selection for its
lifetime, including if the environment subsequently changes.

Public plans, bindings, and output tensors use the same contracts for either
transport. PLE retains `table_memory="io_uring"` as its disk-placement selector.
The environment variable selects the underlying transport. No vLLM setting is
required. PLE's `weight_host` and `weight_scale_host` inspection attributes are
`None` with GDS. Engram's explicitly requested `resident_scales=True` still
retains raw E8M0 scales in mapped host memory.

## Requirements

GDS is optional. Its native module builds only when selected, using a C
compiler, CUDA runtime development files, and cuFile development files with
the parameter APIs introduced in version 1.14. Diagnostics using cuFile's
statistics APIs additionally require version 1.15. `CUDA_HOME` selects the
build toolkit; otherwise the loader uses `pkg-config cufile`, then
`/usr/local/cuda`. The dynamic loader can reuse a cuFile library already loaded
by PyTorch; `CUDA_HOME` does not replace that library. Build identity includes
source, shared headers, toolkit headers, libraries, compiler, and flags.

The source filesystem, storage stack, and assigned GPU must support a direct
GDS path to use GPU transport; cuFile otherwise uses CPU compatibility reads.
Source files remain immutable and are opened read-only with
`O_DIRECT`. Register arbitrary tensor byte offsets directly; no checkpoint
repacking or padding is required. On x86 NVMe RAID, use the `nvidia-fs` route.

The backend defaults `properties.allow_compat_mode` to `true` before loading
cuFile, preserving explicit settings from `CUFILE_ENV_PATH_JSON` or
`/etc/cufile.json` in a private configuration. cuFile selects the transport;
b12x keeps the same device staging and gather path. Explicit cuFile settings and
an already initialized driver remain authoritative. b12x does not modify the
source configuration or close another library's driver. Forced compatibility
is supported through `CUFILE_FORCE_COMPAT_MODE=true`.

## Staging and lifetime

Both transports share 64-bit source and row-offset arithmetic, block
deduplication, and adjacent-block coalescing. Read requests cover aligned 4 KiB
blocks and coalesce up to 64 KiB. Duplicate and shuffled row IDs preserve
compact output order; invalid and nonlocal IDs produce zero rows. Valid
partial reads at EOF require the exact expected byte count. Unexpected
truncation fails.

GDS allocates one device arena of
`min(queue_depth, cuFile batch limit) * 64 KiB`, with alignment padding, and
registers its nonoverlapping 64 KiB slots once. Batch entries use each slot's
registered base with zero device offset. Compact device weight and scale
planes, host IDs, and host/device gather descriptors have capacity fixed at
construction. Two alternating batches share that capacity, or one when the
queue depth is one. Submission overlaps the other batch's reads. Each batch
checks every completion cookie, status, and byte count, then gathers bytes on
the GPU. Its slots are reused only after that gather completes. Gather
descriptors are uploaded once per transaction. Direct GDS payloads stay on the GPU;
compatibility reads use cuFile's CPU staging before reaching the device arena.
ID production still requires the existing GPU-to-CPU transfer for CPU planning.

The byte-gather program is part of each disk component's preparation inventory.
Its specialization is independent of live row and fragment counts. The native
reader launches the precompiled CUDA function after checking its parameter ABI,
launch geometry, and absence of scratch allocations. Disk
preparation runs outside CUDA graphs; downstream graphs consume the stable
BF16 output. Cache transactions serialize reuse across streams, and explicit
close waits for queued consumers. If transport completion cannot be proven,
the reader becomes unusable and retains registered DMA storage until process
exit. Successfully submitted batches are drained without cancellation, including
after another batch reports a short read. Cancellation alone never authorizes
freeing storage.

`stats()` reports requested and physical read bytes, calls, coalescing, planning
and native execution time, host/device staging sizes, and `gds_enabled`.
`gds_enabled` identifies the cuFile backend, including CPU compatibility reads.
`owned_host_bytes` includes host metadata and, for Engram, resident scales.
The process-wide cuFile statistics diagnostics distinguish NVFS/P2P operations
from POSIX/AIO/io_uring compatibility operations. Those counters count batch
entries; `submit_calls` counts batch API submissions. GDS also reports time
inside batch submission and completion queries.

## Qualification

Run the same GPU tests for each backend. For GDS, place pytest's temporary
files on the filesystem being qualified:

```bash
B12X_DISK_BACKEND=gds python -m pytest \
  tests/sequence/test_disk_row_cache.py \
  tests/sequence/test_engram.py tests/sequence/test_ple_embedding.py \
  -k 'disk or file_offsets or source_and_read or backend_is_fixed or close_waits' \
  --basetemp=/path/on/gds-filesystem/b12x-tests
```

The tests cover exact bytes, quantized output, large offsets, truncation,
multiple tables, cross-stream reuse, frozen resolution, and output graph
replay. The cuFile backend requires its development files and runtime libraries;
an unavailable direct GDS route uses CPU compatibility unless explicitly disabled.

`benchmarks/benchmark_ple_disk.py --backends io_uring,gds` measures the public
prepared PLE path on identical physical temporary files and changing queries.
It reverses backend order on alternating query/repeat pairs, checks decoded
outputs, flushes GPU L2 outside timing, and preserves per-transaction metrics.
The default geometry writes 26.82 GiB. Use `--directory` on the target
filesystem and `--output` outside the repository to preserve raw evidence.
The reported ratio is **io_uring latency / GDS latency**; greater than one
favors GDS. GDS remains opt-in regardless of benchmark results.

`benchmarks/benchmark_ngram_ssd.py --models engram --backends io_uring,gds`
compares the prepared hash, disk read, and GPU decode transaction on actual
checkpoint rows. Add `--engram-resident-scales` to retain the E8M0 scale plane
in mapped host memory for both transports. The benchmark records checkpoint
identity, native objects, source hashes, loaded cuFile libraries, physical GPU
identity, raw samples, and direct-transport counters. It checks sampled decoded
rows against CPU file reads for each changing query. The result excludes model
inference, projection, and TP collectives.

Mapped-scale memory is eight bytes per Engram table row, partitioned across TP
ranks. For DeepSeek-V4.1-Flash checkpoint
`fb2764a5cf321eaa5070ca8f9e892818f477c16d`, the tables at layers 1 and 14 contain
384,006,168 and 384,016,682 rows. Their combined scale storage is 5.72 GiB across
TP=4, approximately 1.43 GiB per rank, plus bounded staging and metadata.

API and platform contracts: [NVIDIA cuFile API reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html),
[NVIDIA GDS best practices](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html),
[NVIDIA GDS troubleshooting and RAID support](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html).
