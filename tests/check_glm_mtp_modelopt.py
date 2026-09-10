import json
from pathlib import Path
from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
from vllm.model_executor.model_loader.utils import configure_quant_config
raw=json.loads(Path('/carrier-config.json').read_text())['quantization_config']
config=ModelOptMixedPrecisionConfig.from_config(raw)
prefix='model.layers.45.mlp.experts'
assert config._resolve_quant_algo(prefix) is None
before=dict(config.quantized_layers)
configure_quant_config(config,Glm5NextMTP)
assert config._resolve_quant_algo(prefix)=='MXFP8'
assert config.quantized_layers[prefix]=={'group_size':32,'quant_algo':'MXFP8'}
assert len(config.quantized_layers)==len(before)==43
assert all(config.quantized_layers[k.replace('model.language_model.','model.',1)]==v for k,v in before.items())
print(json.dumps({'status':'passed','before_mtp_algo':None,'after_mtp_algo':config._resolve_quant_algo(prefix),'preserved_layer_records':len(before),'scope':'actual installed configure_quant_config with patched GLM MTP class and pinned carrier metadata; no GPU inference'}))
