"""Native prequantized input must preserve exact expert route outputs."""
import pytest
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from b12x.moe.fused_moe import _impl as moe
from tests.moe.test_v41_expert_numerics import _check_v41_native_expert_routing_and_graph


@pytest.mark.parametrize("m", [1, 16, 80])
def test_prequantized_native_input(m, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        pytest.skip("native Spark prequantized input qualification requires SM121")
    original = moe._get_dynamic_kernel
    state = {"wire": None, "enabled": False}

    def get(*args, **kwargs):
        baseline, mac = original(*args, **kwargs)
        candidate, candidate_mac = original(*args, **kwargs, prequantized_input=True)
        assert candidate_mac == mac

        def run(*launch):
            if state["enabled"]:
                wire = state["wire"]
                assert wire is not None
                pointer = make_ptr(cutlass.Uint8, wire.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                return candidate(pointer, *launch[1:])
            return baseline(*launch)
        return run, mac

    monkeypatch.setattr(moe, "_get_dynamic_kernel", get)

    def check(binding, x, ids, routing):
        h, experts = 5120, 384
        # Encoding is outside the measured consumer, and preserves each row's
        # K32 scales. Actual transport/RTX fusion is not part of this probe.
        baseline = binding.run_route_partials().clone()
        blocks = x.float().reshape(m,h//32,32)
        exponent = torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448))
        scales = torch.exp2(exponent)
        payload = (blocks/scales[...,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(m,h)
        wire = torch.cat([payload,(exponent+127).to(torch.uint8)],dim=1).contiguous()
        saved_x=x.clone();x.zero_()
        state['wire']=wire;state['enabled']=True
        result=binding.run_route_partials()
        assert torch.equal(result,baseline), (m,(result-baseline).abs().max().item())
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=binding.run_route_partials()
        before=torch.cuda.memory_allocated();graph.replay()
        assert torch.cuda.memory_allocated()==before
        assert torch.equal(result,baseline)
        # Poison payload, change routes, and compare against a separately executed
        # BF16-input baseline. Restore the encoded values in the same allocation.
        state['enabled']=False
        x.copy_(saved_x).mul_(-0.5);ids.add_(1).remainder_(experts)
        changed=binding.run_route_partials().clone()
        blocks=x.float().reshape(m,h//32,32)
        exponent=torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4)/448))
        wire[:,:h].copy_((blocks/torch.exp2(exponent)[...,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(m,h))
        wire[:,h:].copy_((exponent+127).to(torch.uint8))
        x.zero_()
        graph.replay()
        assert torch.equal(result,changed),(m,(result-changed).abs().max().item())
        wire[:,:h].zero_();wire[:,h:].fill_(105)
        graph.replay();assert torch.count_nonzero(result)==0
        state['enabled']=False;state['wire']=None

    _check_v41_native_expert_routing_and_graph(m, 576, 384, check)
