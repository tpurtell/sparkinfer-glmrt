"""Load only hash-pinned, inspected PE definitions; never import vendor packages."""
from __future__ import annotations
import ast
import functools
import hashlib
import os
from pathlib import Path
import types

REVISION = "d664cd1e96052857aea2aa844eeba571c24758b4"
SOURCES = {
    "vision.py":"a3ed563be68b63ee47c4103116bb4f77015eaaea31b040d9301b82af85349d16",
    "step_vl.py":"348a7e4d5260d68232d91fe13a3b1a9a6badca4799457f1b7c8a1a8a112adcb2",
    "activation.py":"ad0c12977ac901c38bffd94b0068236a639647e5f65e9c1b78d97e30d27d590a",
    "test_vision.py":"1eddd2ebd456927558a2fd3217335c35d2b3ef2dcbf9af8b76748cf592622f97",
}


def _tree(name):
    root = Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])
    raw = (root/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCES[name]:
        raise ValueError(f"PE vendor source hash mismatch: {name}")
    return ast.parse(raw,filename=str(root/name))


@functools.cache
def inkling():
    import torch
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as tldevice
    names = {"_rope_inv_freq","_bf16","_layer_norm","_rotate_pair","_rotary_qk_kernel",
        "apply_rotary_qk_","_scaled_residual_norm_kernel","scaled_residual_norm_"}
    module = types.ModuleType("perception_vendor_inkling")
    module.__file__ = str(Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"vision.py")
    import sys
    sys.modules[module.__name__] = module
    namespace = module.__dict__
    namespace.update(torch=torch,triton=triton,tl=tl,tldevice=tldevice,functools=functools)
    nodes = [node for node in _tree("vision.py").body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("incomplete pinned Inkling PE definitions")
    # Triton inspects source by filename; write no generated source and execute
    # the original AST nodes with the original source filename/line positions.
    filename = str(Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"vision.py")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,"exec"),namespace)
    return types.SimpleNamespace(**namespace)


@functools.cache
def step5():
    import torch
    from torch import nn
    from einops import rearrange,repeat
    names = {"rotate_half","apply_rotary_emb","PerceptionEncoderRope2D","PerceptionEncoderLayerScale"}
    nodes = [node for node in _tree("step_vl.py").body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("incomplete pinned Step PE definitions")
    namespace = dict(torch=torch,nn=nn,rearrange=rearrange,repeat=repeat)
    filename = str(Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"step_vl.py")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,"exec"),namespace)
    quick = next(node for node in _tree("activation.py").body if isinstance(node,ast.ClassDef) and node.name=="QuickGELU")
    method = next(node for node in quick.body if isinstance(node,ast.FunctionDef) and node.name=="forward_native")
    exec(compile(ast.Module(body=[method],type_ignores=[]),filename,"exec"),namespace)
    return types.SimpleNamespace(**namespace)


def merge(hidden,height,width,channels,class_token):
    """Execute the exact pooling statements in the pinned vendor eager oracle."""
    function = next(node for node in _tree("test_vision.py").body if isinstance(node,ast.FunctionDef) and node.name=="_reference_image")
    nodes = [node for node in function.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=="pooled" for target in node.targets)]
    if len(nodes) != 2:
        raise ValueError("pinned PE pooling statements changed")
    # The vendor definition assumes CLS; prepend a dummy only for the generic
    # no-CLS geometry so the same inspected expressions remain unchanged.
    if not class_token:
        import torch
        hidden = torch.cat((torch.zeros_like(hidden[:1]),hidden))
    namespace = dict(hidden=hidden,h=height//2,w=width//2,m=2,width=channels)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),"vendor_pooling","exec"),namespace)
    return namespace["pooled"]


def metrics(name,actual,reference):
    import torch
    a,r = actual.contiguous().view(torch.int16).int(),reference.contiguous().view(torch.int16).int()
    # Monotonic ordered BF16 encodings, including the negative half interval.
    a = torch.where(a<0,-32768-a,a);r = torch.where(r<0,-32768-r,r)
    ulp = (a-r).abs()
    print(f"PE_ORACLE {name} vendor={REVISION} bit_equal={float((actual==reference).float().mean()):.9f} max_ulp={int(ulp.max())}",flush=True)
    return int(ulp.max())
