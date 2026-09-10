import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("mtp_patch", Path(__file__).resolve().parents[1] / "scripts/patch_glm_mtp_modelopt.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PatchTests(unittest.TestCase):
    source = "from vllm.model_executor.models.utils import maybe_prefix\nclass Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):\n    pass\n"

    def test_scoped_mapper(self):
        result = module.patch(self.source)
        self.assertIn('"model.language_model.": "model."', result)
        self.assertIn('"language_model.model.": "model."', result)
        self.assertIn("hf_to_vllm_mapper = WeightsMapper", result)
        with self.assertRaises(ValueError):
            module.patch(result)

    def test_source_drift_rejected(self):
        with self.assertRaises(ValueError):
            module.patch(self.source.replace("Glm5NextMTP", "UnknownMTP"))


if __name__ == "__main__":
    unittest.main()
