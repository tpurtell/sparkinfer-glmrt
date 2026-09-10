"""TP2 preserves physical trellis words, scale roles, and global sign slicing."""
import unittest
import torch
from b12x.moe._shared.trellismx.p8_tp2_repack import join_tensor
from b12x.moe._shared.trellismx.p8_coupled_scales import rank_local_coupled_signs


class RepackTests(unittest.TestCase):
    def test_trellis_words_reconstruct_global_partition(self):
        # Full logical planes, split on the encoder's TP dimension.
        for name, shape, axis in [('w13_trellis', (2, 3, 8, 16, 80), 3),
                                  ('w2_trellis', (3, 16, 8, 80), 1),
                                  ('w2_scale_ue8m0', (3, 64, 8), 2)]:
            global_tensor = torch.arange(torch.tensor(shape).prod()).reshape(shape)
            tp4 = global_tensor.chunk(4, axis)
            for rank in (0, 1):
                got = join_tensor(name, tp4[2*rank], tp4[2*rank+1])
                expected = global_tensor.chunk(2, axis)[rank]
                self.assertTrue(torch.equal(got, expected), name)

    def test_role_planes_do_not_interleave_parent_ranks(self):
        # Every role and expert has distinct values; simple parent concatenation
        # would silently reorder gate/up or gate_svh/up_svh/down_suh.
        for name, roles in [('w13_scale_ue8m0', 2), ('intermediate_scales_fp16', 3)]:
            full = torch.arange(3 * roles * 32 * 4).reshape(3, roles, 32, 4)
            parents = [x.reshape(3, roles*8, 4) for x in full.chunk(4, 2)]
            for rank in (0, 1):
                expected = full[:, :, rank*16:(rank+1)*16].reshape(3, roles*16, 4)
                self.assertTrue(torch.equal(join_tensor(name, *parents[rank*2:rank*2+2]), expected))

    def test_replicated_tensors_must_match(self):
        for name in ('gate_up_suh_fp16', 'down_svh_fp16', 'coupled_sign_draw_u8'):
            a = torch.ones(16)
            self.assertTrue(torch.equal(join_tensor(name, a, a), a))
            with self.assertRaises(ValueError):
                join_tensor(name, a, a + 1)
        with self.assertRaises(ValueError):
            join_tensor('unrecognized', torch.ones(2), torch.ones(2))

    def test_signs_preserve_global_atom_order(self):
        for rank in (0, 1):
            a, b = [rank_local_coupled_signs(intermediate=512, rank=r)
                    for r in (2*rank, 2*rank+1)]
            expected = torch.cat((a[:1024], b[:1024], a[1024:], b[1024:]))
            actual = rank_local_coupled_signs(intermediate=1024, rank=rank, world_size=2)
            self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
