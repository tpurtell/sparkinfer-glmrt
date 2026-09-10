import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('export_runtime', Path(__file__).resolve().parents[1]/'scripts/export_trellismx_runtime.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)

class ExportTests(unittest.TestCase):
    def test_namespace_and_operator_isolation_preserves_attribution_and_math(self):
        source = '''"""Original b12x documentation: https://github.com/lukealonso/b12x"""
# Copyright b12x contributors
from b12x._lib import compiler
x = torch.ops.b12x.example(2 * y + 3)
operator = "b12x::example"
module = "b12x.moe.fused_moe"
cache = "B12X_COMPILE_CACHE_DIR"
'''
        result = exporter.rewrite_source(source)
        self.assertIn('https://github.com/lukealonso/b12x', result)
        self.assertIn('# Copyright b12x contributors', result)
        self.assertIn('from trellismx_b12x._lib import compiler', result)
        self.assertIn('torch.ops.trellismx_b12x.example(2 * y + 3)', result)
        self.assertIn('trellismx_b12x::example', result)
        self.assertIn('trellismx_b12x.moe.fused_moe', result)
        self.assertIn('TRELLISMX_COMPILE_CACHE_DIR', result)

    def test_manifest_binds_original_and_exported_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source';source.mkdir()
            (source/'__init__.py').write_text('import b12x\n')
            result = exporter.export(source, root/'output')
            record = result['files']['__init__.py']
            self.assertNotEqual(record['source_sha256'], record['generated_sha256'])
            self.assertEqual((root/'output/trellismx_b12x/__init__.py').read_text(), 'import trellismx_b12x\n')

if __name__ == '__main__': unittest.main()
