"""Preparation must preserve runtime dtype keys and the launcher's string ABI."""
from contextlib import nullcontext

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_compile_roce_uses_launcher_dtype_name(monkeypatch, dtype):
    from b12x.comm.roce import _allgather_cute, _oneshot_cute, _preparation

    def launcher(dtype_name, *args):
        # Exercise the real constructor's supported dtype validation without CUDA.
        _oneshot_cute._RoceOneshotLaunch(dtype_name, *args[:-1])
        return dtype_name

    monkeypatch.setattr(_oneshot_cute, "get_launcher", launcher)
    monkeypatch.setattr(_allgather_cute, "get_launcher", lambda *args: "gather")
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: nullcontext())
    payload = dict(
        surface="AllReduce.all_reduce", world_size=2, rank=0,
        topology="roce_rdma", peer_hosts=("rhea", "moa"),
        hca_names=("roceP2p1s0f0",), call={"dtypes": (str(dtype).split(".")[-1],)},
        setup={"threads": 256, "slots": 6, "flag_stride": 16, "hca_count": 1},
    )
    programs = _preparation.compile_roce(payload, 0)
    assert set(programs) == {dtype, "gather"}
    assert programs[dtype] == str(dtype).split(".")[-1]
