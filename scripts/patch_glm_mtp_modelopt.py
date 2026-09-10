"""Install the GLM MTP multimodal quantization-prefix mapper in pinned vLLM."""
from pathlib import Path
import sys

MARKER = "# GLM MTP carrier quantization namespace"


def patch(source: str) -> str:
    if MARKER in source:
        raise ValueError("GLM MTP quantization mapper is already installed")
    import_anchor = "from vllm.model_executor.models.utils import maybe_prefix"
    class_anchor = "class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):\n"
    if source.count(import_anchor) != 1 or source.count(class_anchor) != 1:
        raise ValueError("Unsupported GLM MTP source; inspect before patching")
    source = source.replace(import_anchor, import_anchor.replace("maybe_prefix", "WeightsMapper, maybe_prefix"))
    mapper = '''    # GLM MTP carrier quantization namespace
    # configure_quant_config applies this before constructing draft layers.
    # Parameter loading already strips these wrappers independently.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "model.language_model.": "model.",
        "language_model.model.": "model.",
    })

'''
    source = source.replace(class_anchor, class_anchor + mapper)
    compile(source, "glm_mtp.py", "exec")
    return source


if __name__ == "__main__":
    path = Path(sys.argv[1])
    path.write_text(patch(path.read_text()))
    print(f"Installed GLM MTP quantization mapper: {path}")
