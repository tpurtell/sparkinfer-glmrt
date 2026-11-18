"""Untimed diagnostic for the 656-byte context-partial row-56 discrepancy.

Run as a script in the coordinator image with the candidate b12x on PYTHONPATH.
--bench is the organizer's seeded context benchmark (used only to build inputs).
No timing loop. Numeric records go to JSONL; a failing verdict exits nonzero.
"""
import argparse
import faulthandler
import importlib.util
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from cuda.bindings import driver as cuda

from b12x.integration.cuteafd import GLM53, context_split as cs, mistral_large4


CAP = 64
WATCH = [55, 56, 57, 58, 59, 60, 61, 62, 63]


def independent(cache, q, ids, begin, scale):
    # Decode physical bytes on the CPU without the benchmark unpacker. Use
    # FP64 row-by-row GEMMs, not its GPU batches of eight. Return both ranks
    # and the direct unsharded result; no kernel merge enters the oracle.
    rows = cache.cpu().contiguous().view(-1, 656)
    codes = rows[:, :512].contiguous().view(torch.float8_e4m3fn).double()
    scales = rows[:, 512:528].contiguous().view(torch.float32).double()
    v = (codes.view(-1, 4, 128) * scales[:, :, None]).view(-1, 512)
    rope = rows[:, 528:].contiguous().view(torch.bfloat16).double()
    k = torch.cat([v, rope], dim=1)
    qq, ii = q[WATCH, begin:begin+32].cpu().double(), ids[WATCH].cpu().long()
    refs = [([], []) for _ in range(3)]
    for query, idx in zip(qq, ii):
        score = (query @ k[idx].T) * scale
        for (outputs, lses), sl in zip(refs, [slice(0, 1536), slice(1536, 2048), slice(None)]):
            x = score[:, sl]
            outputs.append(x.softmax(-1) @ v[idx[sl]])
            lses.append(x.logsumexp(-1))
    return [(torch.stack(o), torch.stack(l)) for o, l in refs]


def row_errors(actual, expected):
    error = (actual.double() - expected).square().mean((-1, -2)).sqrt()
    norm = expected.square().mean((-1, -2)).sqrt()
    return (error / norm).tolist()


def run(args, emit):
    # Keep markers off the numeric JSONL contract. The last marker plus
    # Python's fatal stack identifies host faults before a partial exists.
    def phase(name):
        print(f"row56 phase: {name}", file=sys.stderr, flush=True)

    phase("import benchmark fixture module")
    spec = importlib.util.spec_from_file_location('seeded_context_bench', args.bench)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    torch.backends.cuda.matmul.allow_tf32 = False
    phase("set CPU thread count")
    torch.set_num_threads(4)
    phase("create CUDA stream")
    stream = torch.cuda.Stream()
    handle = cuda.CUstream(stream.cuda_stream)
    bad = 0
    for family in args.cells.split(','):
        g = {'glm': GLM53, 'ml4': mistral_large4.rope_attention()}[family]
        phase(f"{family}: build seeded fixture")
        gen = torch.Generator(device='cuda').manual_seed(bench.aot_cell_seed(41512, family, 'asymmetric'))
        cache, idx, _, mk, mv = bench.make_records(2048, 64, 576, False, gen, CAP, 'cuda')
        q = (torch.randn(CAP, 64, 576, generator=gen, device='cuda') * 2.5).bfloat16()
        local = []
        for sl in [slice(0, 1536), slice(1536, 2048)]:
            owned = idx[:, sl]
            table = torch.full_like(idx, -1)
            table[:, :owned.shape[1]] = owned
            local.append((table, torch.full((CAP,), owned.shape[1], device='cuda', dtype=torch.int32)))
        phase(f"{family}: compile combine (no launch)")
        combine = cs.compile_lse_combine2_aot(heads=32, has_sink=False)
        for begin in args.head_begin:
            phase(f"{family} head={begin}: CPU FP64 reference")
            refs = independent(cache, q, idx, begin, g.softmax_scale)
            # Recompute exactly the benchmark output reference, including the
            # last batch at lo=56, and retain its original CPU snapshot.
            parts = []
            for lo in range(0, CAP, 8):
                phase(f"{family} head={begin}: torch BMM reference lo={lo}")
                ids = idx[lo:lo+8].long()
                k, v = mk[ids], mv[ids]
                score = torch.bmm(q[lo:lo+8, begin:begin+32].float(), k.transpose(1, 2)) * g.softmax_scale
                parts.append(torch.bmm(score.softmax(-1), v))
            batched = torch.cat(parts)
            phase(f"{family} head={begin}: synchronize and copy reference")
            frozen = batched[WATCH].cpu()
            ref_err = row_errors(frozen, refs[2][0])
            emit(dict(stage='reference', family=family, begin=begin, watch=WATCH, bmm_vs_cpu=ref_err))
            bad += not all(x < 1e-4 for x in ref_err)
            for splits in args.splits:
                for arm in ('fp8', 'bf16'):
                    phase(f"{family} head={begin} {splits=} {arm}: compile partial")
                    program = cs.compile_sparse_mla_partial_aot(g, max_rows=CAP, head_begin=begin,
                                head_count=32, num_splits=splits, precision=arm)
                    scratch = [torch.empty(program.scratch_bytes(CAP)['scratch'], device='cuda', dtype=torch.uint8)
                               for _ in local]
                    outs = [torch.empty((CAP, 32, 512), device='cuda', dtype=torch.bfloat16) for _ in local]
                    lses = [torch.empty((CAP, 32), device='cuda') for _ in local]
                    combined = torch.empty_like(outs[0])
                    eager_hashes = {}
                    for n in args.rows:
                        def partial(rank):
                            table, lens = local[rank]
                            program.launch(q, cache, table, lens, None, None, None, outs[rank],
                                           lses[rank], scratch[rank], scalars=(n,), stream=handle)

                        def reset():
                            for rank in (0, 1):
                                # All-ones bytes poison BF16 data and FP32 LSE.
                                scratch[rank].fill_(255)
                                outs[rank].fill_(float('nan'))
                                lses[rank].fill_(float('nan'))
                            combined.fill_(float('nan'))

                        phase(f"{family} head={begin} {splits=} {arm} rows={n}: warm and capture")
                        torch.cuda.synchronize()
                        with torch.cuda.stream(stream):
                            for rank in (0, 1):
                                partial(rank)
                        stream.synchronize()
                        graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
                        for rank in (0, 1):
                            with torch.cuda.graph(graphs[rank], stream=stream):
                                partial(rank)
                        count = sum(r < n for r in WATCH)
                        observed = []
                        for mode in ('eager-poison', 'replay-poison', 'replay-dirty', 'replay-poison-again'):
                            phase(f"{family} head={begin} {splits=} {arm} rows={n}: {mode}")
                            with torch.cuda.stream(stream):
                                if mode != 'replay-dirty':
                                    reset()
                                for rank in (0, 1):
                                    if mode == 'eager-poison':
                                        partial(rank)
                                    else:
                                        graphs[rank].replay()
                                combine.launch(outs[0], lses[0], outs[1], lses[1], None, combined,
                                               scalars=(n,), stream=handle)
                            stream.synchronize()
                            data = [a[WATCH[:count]].cpu() for a in [*outs, combined]]
                            ls = [a[WATCH[:count]].cpu() for a in lses]
                            err = [row_errors(a, r[0][:count]) for a, r in zip(data, refs)]
                            lerr = [(a.double()*math.log(2)-r[1][:count]).abs().amax(-1).tolist()
                                    for a, r in zip(ls, refs)]
                            whole_finite = all(bool(a[:n].isfinite().all()) for a in [*outs, *lses, combined])
                            guard = all(bool(a[n:].isnan().all()) for a in [*outs, *lses, combined])
                            repeated = True
                            if not observed:
                                observed = data + ls
                                h = hashlib.sha256()
                                for a in observed:
                                    h.update(a.contiguous().view(torch.uint8).numpy().tobytes())
                                digest = h.hexdigest()
                                repeated = digest == eager_hashes.setdefault(n, digest)
                            same = [torch.equal(a, b) for a, b in zip(observed, data+ls)]
                            ref_same = torch.equal(batched[WATCH].cpu(), frozen)
                            rec = dict(family=family, begin=begin, splits=splits, arm=arm, rows=n, mode=mode,
                                       watch=WATCH[:count], rank0_rms=err[0], rank1_rms=err[1], combined_rms=err[2],
                                       rank_lse_abs=lerr, finite=whole_finite, inactive_untouched=guard,
                                       same_as_eager=same, repeated_eager_match=repeated, ref_unchanged=ref_same)
                            emit(rec)
                            limit = .15 if arm == 'fp8' else .02
                            bad += not (all(x < limit for group in err for x in group) and all(same) and
                                        whole_finite and guard and ref_same and repeated)
                        for graph in graphs:
                            graph.reset()
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench', required=True, type=Path)
    p.add_argument('--jsonl', required=True, type=Path)
    p.add_argument('--cells', default='glm,ml4')
    p.add_argument('--splits', type=int, nargs='+', default=[32, 11, 4])
    p.add_argument('--rows', type=int, nargs='+', default=[56, 57, 64, 57])
    p.add_argument('--head-begin', type=int, nargs='+', choices=[0, 32], default=[0])
    args = p.parse_args()
    faulthandler.enable(all_threads=True)
    if not all(56 <= x <= CAP for x in args.rows):
        p.error('this probe watches rows 55..63; use live counts in 56..64')
    with args.jsonl.open('w') as out:
        def emit(rec):
            print(json.dumps(rec), file=out, flush=True)
        bad = run(args, emit)
        emit(dict(stage='verdict', failures=bad))
    print(f'{args.jsonl}: {bad} failing diagnostic records')
    return int(bad != 0)


if __name__ == '__main__':
    sys.exit(main())
