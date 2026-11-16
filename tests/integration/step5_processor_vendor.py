"""Execute the hash-pinned Step5 registered processing-info construction path.

No vLLM model code is imported. Only inspected vendor classes and the actual
processing-info factory methods execute; the Step5->Step3p7->Step3VL hierarchy
and processor registry decorators are validated from the immutable source AST.
"""
from __future__ import annotations
import ast
import __future__
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

PINS={
    "step3_processor.py":"ce0265261a6bffaf40b98911ef5b9babe30d20443af83a4c8ecf251ce6a861fa",
    "step3_vl.py":"323acae3da8e5ede13a31d218ea95550a0890186b3d226e8ca81c4778949180a",
    "step3p7.py":"c6585739153a78bc2ac3aa1655770dd4b1f1317a73a341c8c21a966e8b035174",
    "step5_vl.py":"13ac15da587896b8e1dd9c5cc24feabc38171a1f902236038ac7148940912bd8",
}


def tree(name):
    p=Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/name;raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PINS[name]:raise ValueError(f"Step processor source changed:{name}")
    return ast.parse(raw,filename=str(p))


def construct(config,tokenizer):
    from itertools import product
    from math import ceil
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms.functional import InterpolationMode
    from transformers import BatchFeature,ImageProcessingMixin,ProcessorMixin,TensorType
    five=next(n for n in tree("step5_vl.py").body if isinstance(n,ast.ClassDef) and n.name=="Step5ForConditionalGeneration")
    seven=next(n for n in tree("step3p7.py").body if isinstance(n,ast.ClassDef) and n.name=="Step3p7ForConditionalGeneration")
    if ast.unparse(five.bases[0])!="Step3p7ForConditionalGeneration" or ast.unparse(seven.bases[0])!="Step3VLForConditionalGeneration":
        raise ValueError("Step5 multimodal inheritance changed")
    third=tree("step3_vl.py")
    model=next(n for n in third.body if isinstance(n,ast.ClassDef) and n.name=="Step3VLForConditionalGeneration")
    registered=[ast.unparse(d) for d in model.decorator_list]
    if not any("Step3VLProcessingInfo" in d and "Step3VLMultiModalProcessor" in d for d in registered):
        raise ValueError("Step5 inherited multimodal registry changed")
    namespace=dict(product=product,ceil=ceil,np=np,torch=torch,Image=Image,transforms=transforms,
        InterpolationMode=InterpolationMode,BatchFeature=BatchFeature,ImageProcessingMixin=ImageProcessingMixin,
        ProcessorMixin=ProcessorMixin,TensorType=TensorType,TokenizerLike=object,
        MAX_IMAGE_SIZE=3024,ImageWithPatches=tuple,__name__="step5_vendor_processor")
    classes={"Step3VisionProcessor","ImagePatcher","Step3VLImageProcessor","Step3VLProcessor"}
    nodes=[n for n in tree("step3_processor.py").body if isinstance(n,ast.ClassDef) and n.name in classes]
    filename=str(Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"step3_processor.py")
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,"exec",flags=__future__.annotations.compiler_flag),namespace)
    info=next(n for n in third.body if isinstance(n,ast.ClassDef) and n.name=="Step3VLProcessingInfo")
    methods=[n for n in info.body if isinstance(n,ast.FunctionDef) and n.name in {"get_image_processor","get_hf_processor"}]
    exec(compile(ast.Module(body=methods,type_ignores=[]),str(Path(os.environ["CUTEAFD_PE_VENDOR_DIR"])/"step3_vl.py"),"exec",flags=__future__.annotations.compiler_flag),namespace)
    class Info:
        def get_hf_config(self):return config
        def get_tokenizer(self):return tokenizer
    Info.get_image_processor=namespace["get_image_processor"]
    Info.get_hf_processor=namespace["get_hf_processor"]
    selected=Info()
    processor=selected.get_hf_processor()
    return processor,namespace
