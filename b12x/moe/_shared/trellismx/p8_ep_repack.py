"""Hash-bound TP4-to-EP2/EP4 compressed weight views, without requantization.

Every owned expert keeps all four intermediate partitions. Slice the expert
axis before joining, so loading one EP rank never materializes all experts at
full intermediate width. Kernels and distributed routing need separate gates.
"""
from contextlib import ExitStack, contextmanager
import torch
from safetensors import safe_open
from .p8_tp2_repack import file_sha256, join_tensor
from .p8_coupled_scales import (
    SCALE_NAMES, rank_local_coupled_signs, tensor_sha256, validate_coupled_component,
)


def expert_partition_tensor(name, parents, begin, end):
    if len(parents) != 4 or not 0 <= begin < end:
        raise ValueError('An expert partition requires four TP4 parents and a valid range')
    axes = {'w13_trellis': 1, 'w2_trellis': 0, 'w13_scale_ue8m0': 0,
            'w2_scale_ue8m0': 0, 'intermediate_scales_fp16': 0}
    if name in axes:
        axis = axes[name]
        if any(p.shape[axis] < end for p in parents):
            raise ValueError('Expert range exceeds a parent tensor')
        parents = [p.narrow(axis, begin, end-begin) for p in parents]
    elif name not in ('gate_up_suh_fp16', 'down_svh_fp16', 'coupled_sign_draw_u8'):
        raise ValueError(f'Unknown P8 tensor: {name}')
    pairs = [join_tensor(name, *parents[i:i+2]) for i in (0, 2)]
    return join_tensor(name, *pairs)


class _ExpertPartition:
    def __init__(self, parents, hashes, *, layer, ep_rank, ep_size):
        if ep_size not in (2, 4) or ep_rank not in range(ep_size):
            raise ValueError('P8 expert partition requires EP2 or EP4')
        self.parents = parents
        self.begin = ep_rank * (288 // ep_size)
        self.end = self.begin + 288 // ep_size
        metadata = [p.metadata() for p in parents]
        for rank, (parent, meta) in enumerate(zip(parents, metadata)):
            expected = {'schema': 'glm53-p8-coupled-h512-h128-tp4-rank.v1',
                        'layer': str(layer), 'rank': str(rank), 'world_size': '4',
                        'global_intermediate': '2048', 'alphabet': 'e4m3',
                        'scale': 'ue8m0-k32', 'law': 'procedural-mcg-alpha2', 'ldlq': 'false'}
            if any(meta.get(k) != v for k, v in expected.items()):
                raise ValueError('Invalid TP4 parent identity')
            validate_coupled_component(meta, {n: parent.get_tensor(n) for n in SCALE_NAMES},
                layer=layer, rank=rank, experts=288, hidden=4096, intermediate=512,
                expected_transform_sha256=meta['encoder_transform_sha256'])
        for key in ('bits', 'source_design_sha256', 'encoder_transform_sha256'):
            if any(m.get(key) != metadata[0].get(key) for m in metadata[1:]):
                raise ValueError(f'Incompatible TP4 parent {key}')
        self._metadata = {k: v for k, v in metadata[0].items()
            if not k.startswith('sha256_') and not k.endswith('_bpw')
            and k not in ('stored_metadata_bytes', 'runtime_regenerated_sign_bytes')}
        self._metadata.update(schema='glm53-p8-coupled-h512-h128-tp1-rank.v1',
            rank='0', world_size='1', local_atom_begin='0', ep_rank=str(ep_rank),
            ep_size=str(ep_size), expert_begin=str(self.begin), expert_end=str(self.end),
            runtime_regenerated_sign_bytes=str(3 * 2048 * 2))
        for rank, digest in enumerate(hashes):
            self._metadata[f'tp4_parent_sha256_{rank}'] = digest
        for name in SCALE_NAMES:
            self._metadata[f'sha256_{name}'] = tensor_sha256(self.get_tensor(name))
        self._metadata['sha256_coupled_signs_fp16'] = tensor_sha256(
            rank_local_coupled_signs(intermediate=2048, rank=0, world_size=1))

    def metadata(self):
        return dict(self._metadata)

    def get_tensor(self, name):
        return expert_partition_tensor(name, [p.get_tensor(name) for p in self.parents], self.begin, self.end)


@contextmanager
def open_ep_group(paths, hashes, *, layer, ep_rank, ep_size):
    if len(paths) != 4 or len(hashes) != 4:
        raise ValueError('EP requires all four ordered TP4 parent shards')
    if ep_size not in (2, 4) or ep_rank not in range(ep_size):
        raise ValueError('Invalid expert-parallel topology')
    for path, digest in zip(paths, hashes):
        if len(digest) != 64 or file_sha256(path) != digest:
            raise ValueError(f'TP4 parent hash mismatch: {path}')
    with ExitStack() as stack:
        parents = [stack.enter_context(safe_open(path, framework='pt', device='cpu')) for path in paths]
        yield _ExpertPartition(parents, hashes, layer=layer, ep_rank=ep_rank, ep_size=ep_size)
