"""Ordered K16 BF16 MMA with register activations and warp-local weight rings."""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32
from b12x._lib.intrinsics import (
    bf16_mma_m16n8k16_f32, cp_async4_shared_global,
    ld_global_nc_u32, ld_shared_u32, shared_ptr_to_u32,
)


class RegisterABf16Projection:
    def __init__(self, n, k, *, tile_n=16, num_stages=4, tile_k=128,
                 out_dtype=cutlass.BFloat16):
        self.n, self.k = int(n), int(k)
        self.tile_n, self.num_stages, self.tile_k = tile_n, num_stages, tile_k
        self.warps = tile_n // 8
        self.out_dtype = out_dtype
        self.pitch = tile_k * 2 + 16
        self.warp_bytes = 8 * self.pitch
        self.stage_bytes = tile_n * self.pitch
        self.tile_m = 16
        self.buffer_align_bytes = 0
        if n % tile_n or k % tile_k or tile_n % 8 or num_stages not in (2, 4, 8):
            raise ValueError('register-A projection needs complete N/K tiles and 2/4/8 stages')

    def _storage(self):
        class Storage:
            pass
        Storage.__annotations__ = {
            'weights': cute.struct.Align[cute.struct.MemRange[Uint32, self.stage_bytes * self.num_stages // 4], 16],
        }
        return cute.struct(Storage)

    @cute.jit
    def __call__(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer,
                 rows: Int32, stream: cuda.CUstream):
        self.kernel(x, w, out, rows).launch(
            grid=(self.n // self.tile_n, (rows + 15) // 16, 1),
            block=(32 * self.warps, 1, 1), stream=stream)

    @cute.jit
    def _issue(self, base: Int32, w: cute.Pointer, n0: Int32,
               chunk: Int32, lane: Int32):
        for i in cutlass.range_constexpr(self.tile_k // 32):
            vector = lane + Int32(i * 32)
            col = vector // Int32(self.tile_k // 8)
            kv = vector % Int32(self.tile_k // 8)
            dst = base + col * Int32(self.pitch) + kv * Int32(16)
            src = Int64(w.toint()) + (Int64(n0 + col) * Int64(self.k)
                  + Int64(chunk) * Int64(self.tile_k) + Int64(kv * 8)) * Int64(2)
            cp_async4_shared_global(dst, src)

    @cute.kernel
    def kernel(self, x: cute.Pointer, w: cute.Pointer, out: cute.Pointer, rows: Int32):
        tid = Int32(cute.arch.thread_idx()[0])
        warp, lane = tid // 32, tid % 32
        g, j = lane // 4, lane % 4
        n0 = Int32(cute.arch.block_idx()[0]) * self.tile_n + warp * 8
        m0 = Int32(cute.arch.block_idx()[1]) * 16
        storage = utils.SmemAllocator().allocate(self._storage())
        base = shared_ptr_to_u32(storage.weights.data_ptr()) + warp * self.warp_bytes
        acc = cute.make_rmem_tensor((4,), Float32)
        acc.fill(0.0)
        for stage in cutlass.range_constexpr(self.num_stages - 1):
            self._issue(base + Int32(stage * self.stage_bytes), w, n0, Int32(stage), lane)
            cute.arch.cp_async_commit_group()
        for chunk in cutlass.range(self.k // self.tile_k, unroll=1):
            cute.arch.cp_async_wait_group(self.num_stages - 2)
            cute.arch.sync_warp()
            ring = (chunk % self.num_stages) * self.stage_bytes
            next_chunk = chunk + self.num_stages - 1
            if next_chunk < self.k // self.tile_k:
                self._issue(base + (next_chunk % self.num_stages) * self.stage_bytes,
                            w, n0, next_chunk, lane)
            cute.arch.cp_async_commit_group()
            # Masked register loads avoid all zero-row shared-memory/TMA traffic.
            a = cute.make_rmem_tensor((self.tile_k // 4,), Uint32)
            a.fill(Uint32(0))
            for h in cutlass.range_constexpr(self.tile_k // 16):
                for half in cutlass.range_constexpr(2):
                    row = m0 + g + Int32(half * 8)
                    if row < rows:
                        addr = Int64(x.toint()) + (Int64(row) * Int64(self.k)
                               + Int64(chunk) * Int64(self.tile_k) + Int64(h * 16) + Int64(j * 2)) * Int64(2)
                        a[4 * h + half] = ld_global_nc_u32(addr)
                        a[4 * h + 2 + half] = ld_global_nc_u32(addr + Int64(16))
            for h in cutlass.range_constexpr(self.tile_k // 16):
                addr = base + ring + g * self.pitch + j * 4 + Int32(h * 32)
                b0 = ld_shared_u32(addr)
                b1 = ld_shared_u32(addr + Int32(16))
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    acc[0], acc[1], acc[2], acc[3],
                    a[4*h], a[4*h+1], a[4*h+2], a[4*h+3], b0, b1)
                acc[0], acc[1], acc[2], acc[3] = d0, d1, d2, d3
            cute.arch.sync_warp()
        cute.arch.cp_async_wait_group(0)
        for half in cutlass.range_constexpr(2):
            row = m0 + g + Int32(half * 8)
            if row < rows:
                for pair in cutlass.range_constexpr(2):
                    idx = Int64(row) * Int64(self.n) + Int64(n0 + j * 2 + pair)
                    out[idx] = acc[half * 2 + pair].to(self.out_dtype)
