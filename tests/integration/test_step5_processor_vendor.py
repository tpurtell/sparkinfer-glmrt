"""Checkpoint-selected Step5 global/local processor execution audit."""
from __future__ import annotations
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from .step5_processor_vendor import construct


def test_checkpoint_selected_step5_processor_uses_actual_global_and_local_filter():
    from transformers import PreTrainedTokenizerFast
    snapshot=Path(os.environ["CUTEAFD_PE_STEP5_SNAPSHOT"])
    cfg=json.loads((snapshot/"config.json").read_text())
    assert cfg["model_type"]=="step5_vl"
    tokenizer=PreTrainedTokenizerFast(tokenizer_file=str(snapshot/"tokenizer.json"))
    processor,scope=construct(SimpleNamespace(vision_config=SimpleNamespace(**cfg["vision_config"])),tokenizer)
    chosen=processor.image_processor
    filters=[]
    for name in ["transform","patch_transform"]:
        resize=next(t for t in getattr(chosen.image_preprocessor,name).transforms if hasattr(t,"interpolation"))
        filters.append(resize.interpolation.value)
    print(f"STEP5_PROCESSOR filters={filters} global={chosen.image_size} local={chosen.patch_size}",flush=True)
    assert filters==["bilinear","bilinear"]
    files=sorted(Path(os.environ["CUTEAFD_PE_REAL_IMAGES"]).glob("*.png"))
    image=Image.open(files[-1]).convert("RGB")
    # Actual natural fixture already has a long side>728 and triggers crops.
    assert max(image.size)>=728
    result=chosen(image,return_tensors="pt")
    patches=result["num_patches"][0]
    assert patches>0
    raw,crops,newlines=chosen.patcher(image)
    assert patches==len(crops)
    globals=chosen.image_preprocessor(raw)["pixel_values"]
    locals=torch.cat([chosen.image_preprocessor(crop,is_patch=True)["pixel_values"] for crop in crops])
    assert torch.equal(globals,result["pixel_values"])
    assert torch.equal(locals,result["patch_pixel_values"])
    assert tuple(globals.shape[-2:])==(728,728) and tuple(locals.shape[-2:])==(504,504)
    # Execute the registered processor's own replacement method, not a local
    # guess about global-first versus crop-first feature ordering.
    replacement=processor.get_image_repl_feature_ids(1,patches,newlines)
    image_start=tokenizer.convert_tokens_to_ids("<im_start>")
    patch_start=tokenizer.convert_tokens_to_ids("<patch_start>")
    assert replacement[0]==patch_start
    assert replacement[-171]==image_start
    print(f"STEP5_PROCESSOR image={image.size} crop_count={patches} newline_mask={newlines} feature_order=local_then_global global_max={float(globals.max()):.9g} local_max={float(locals.max()):.9g}",flush=True)
