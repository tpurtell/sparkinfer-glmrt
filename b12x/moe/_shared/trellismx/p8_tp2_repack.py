"""Lossless, CPU load-time TP4-to-TP2 view of adjacent coupled P8 shards.

No trellis decoding or requantization occurs here. Native kernels consume the
joined compressed streams. Parent file hashes remain the trust boundary.
"""
from contextlib import ExitStack, contextmanager
import hashlib
from pathlib import Path

import torch
from safetensors import safe_open

from .p8_coupled_scales import (
    SCALE_NAMES, rank_local_coupled_signs, tensor_sha256,
    validate_coupled_component,
)


def join_tensor(name, left, right):
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError(f'Incompatible TP4 tensor pair: {name}')
    if name in ('gate_up_suh_fp16', 'down_svh_fp16', 'coupled_sign_draw_u8'):
        if not torch.equal(left, right):
            raise ValueError(f'Replicated TP4 tensor differs: {name}')
        return left.clone()
    axes = {'w13_trellis': 3, 'w2_trellis': 1, 'w2_scale_ue8m0': 2}
    if name in axes:
        return torch.cat((left, right), dim=axes[name]).contiguous()
    sections = {'w13_scale_ue8m0': 2, 'intermediate_scales_fp16': 3}
    if name in sections:
        count = sections[name]
        if left.shape[1] % count:
            raise ValueError(f'Invalid role dimension: {name}')
        return torch.cat([
            torch.cat((a, b), dim=1)
            for a, b in zip(left.chunk(count, dim=1), right.chunk(count, dim=1))
        ], dim=1).contiguous()
    raise ValueError(f'Unknown P8 tensor: {name}')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class _Pair:
    def __init__(self, parents, hashes, *, layer, rank):
        self.parents = parents
        metadata = [p.metadata() for p in parents]
        for offset, (parent, meta) in enumerate(zip(parents, metadata)):
            expected = {'schema': 'glm53-p8-coupled-h512-h128-tp4-rank.v1',
                        'layer': str(layer), 'rank': str(2 * rank + offset),
                        'world_size': '4', 'global_intermediate': '2048',
                        'alphabet': 'e4m3', 'scale': 'ue8m0-k32',
                        'law': 'procedural-mcg-alpha2', 'ldlq': 'false'}
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError('Invalid TP4 parent identity')
            tensors = {name: parent.get_tensor(name) for name in SCALE_NAMES}
            validate_coupled_component(meta, tensors, layer=layer,
                rank=2 * rank + offset, experts=288, hidden=4096,
                intermediate=512, expected_transform_sha256=meta['encoder_transform_sha256'])
        for key in ('bits', 'source_design_sha256', 'encoder_transform_sha256'):
            if metadata[0].get(key) != metadata[1].get(key):
                raise ValueError(f'Incompatible TP4 parent {key}')
        self._metadata = {key: value for key, value in metadata[0].items()
                          if not key.startswith('sha256_') and not key.endswith('_bpw')
                          and key not in ('stored_metadata_bytes', 'runtime_regenerated_sign_bytes')}
        self._metadata['runtime_regenerated_sign_bytes'] = '6144'
        self._metadata.update(schema='glm53-p8-coupled-h512-h128-tp2-rank.v1',
                              rank=str(rank), world_size='2',
                              local_atom_begin=str(rank * 32),
                              tp4_parent_sha256_0=hashes[0], tp4_parent_sha256_1=hashes[1])
        for name in SCALE_NAMES:
            self._metadata[f'sha256_{name}'] = tensor_sha256(self.get_tensor(name))
        self._metadata['sha256_coupled_signs_fp16'] = tensor_sha256(
            rank_local_coupled_signs(intermediate=1024, rank=rank, world_size=2))

    def metadata(self):
        return dict(self._metadata)

    def get_tensor(self, name):
        return join_tensor(name, *(parent.get_tensor(name) for parent in self.parents))


@contextmanager
def open_tp2_pair(paths, hashes, *, layer, rank):
    if rank not in (0, 1) or len(paths) != 2 or len(hashes) != 2:
        raise ValueError('TP2 requires one adjacent pair of TP4 shards')
    for path, digest in zip(paths, hashes):
        if len(digest) != 64 or file_sha256(path) != digest:
            raise ValueError(f'TP4 parent hash mismatch: {path}')
    with ExitStack() as stack:
        parents = [stack.enter_context(safe_open(path, framework='pt', device='cpu'))
                   for path in paths]
        yield _Pair(parents, hashes, layer=layer, rank=rank)
