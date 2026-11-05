"""Small stdlib-only contract checks; no models, inference, GPU or real data."""
import ast,importlib.util,json,sys,tempfile,unittest,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module

class ContractTests(unittest.TestCase):
    def test_syntax(self):
        for path in ROOT.rglob('*.py'):ast.parse(path.read_text(),str(path))

    def test_manifest_scope(self):
        d=json.loads((ROOT/'WEIGHT_MANIFEST.json').read_text())
        self.assertEqual(d['submission_ID'],'1e7ab392')
        self.assertEqual(len(d['files']),7)
        self.assertFalse(any(x['path'].startswith('fomo/') for x in d['files']))

    def test_synthetic_zip_selection(self):
        runtime=load('w045_contract_runtime',ROOT/'runtime/app/runtime_io.py')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with zipfile.ZipFile(root/'EXAMPLE_STUDY.zip','w') as z:
                for name in ('t1w','flair','t2w'):
                    z.writestr(f'EXAMPLE_STUDY/img/EXAMPLE_STUDY_{name}-raw-axi.nii.gz',b'synthetic-not-NIfTI')
                z.writestr('EXAMPLE_STUDY/atlas/ignored.nii.gz',b'synthetic')
            cases=runtime.discover_cases(root)
            selected=runtime.scan_case_zip(cases[0])
            self.assertEqual(set(selected.selected),{'t1','flair','t2'})
            self.assertEqual(len(runtime.input_inventory_fingerprint(cases)),64)

    def test_weights_not_in_source(self):
        self.assertFalse(any(ROOT.rglob('*.pt')))
        self.assertFalse(any(ROOT.rglob('*.safetensors')))
        self.assertFalse(any(ROOT.rglob('*.npy')))

if __name__=='__main__':unittest.main()
