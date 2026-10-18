import importlib.util,json,os,tempfile,unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('v7pipeline',ROOT/'resources/pipeline.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        p.write(root/'config.json',{'gpu':4,'esm_checkpoint':'cached-esm'})
        p.write(root/'frozen_manifest.json',{'source_pins':{'config.json':p.sha(root/'config.json')}})
        p.write(root/'data/manifest.json',{'summary':{'splits':{},'original_train_rows':378,'original_protected_rows':210},'grouping':'fixed','split_rule':'hash'})
        for name in ('panel.jsonl','catalog.json'):(root/'data'/name).write_text('{}')
        p.write(root/'assets/DOWNLOAD_MANIFEST.json',{'official_repository_commit':'fixed','files':[]})
        old=root/'protected';p.write(old/'frozen_manifest.json',{'files':[]})
        p.write(root/'private/runtime.json',{'protected_v6_scope':str(old),'protected_v6_manifest_sha256':p.sha(old/'frozen_manifest.json')})

    def test_frozen_bytes_and_protected_prior_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            s=Path(directory);self.fixture(s);self.assertTrue(p.protected_v6(s));p.frozen(s)
            (s/'config.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'integrity'):p.frozen(s)
            p.write(s/'protected/frozen_manifest.json',{'files':[],'changed':True})
            with self.assertRaisesRegex(RuntimeError,'Protected'):p.protected_v6(s)

    def test_child_environment_and_complete_finite_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            s=Path(directory);self.fixture(s);environments=[];commands=[];old_cwd=Path.cwd()
            class Child:
                pid=123456
                def __init__(self,args,**kwargs):
                    environments.append(kwargs['env']);commands.append(args)
                    if '--stage' in args and 'extract' in args:p.write(s/'cache/reference_smoke.json',{'passed':True,'max_absolute_error':0.0001})
                def wait(self,timeout):return 0
            try:
                with patch.object(p,'lock',return_value=nullcontext()),patch.object(p.socket,'gethostname',return_value='node1'),patch.object(p.subprocess,'Popen',Child),patch.object(p,'identity',return_value={'pid':123456,'session':123456,'start_ticks':1}),patch.object(p,'gpu_check',return_value=True):
                    result=p.run(s)
            finally:os.chdir(old_cwd)
            self.assertEqual(result['status'],'complete');self.assertEqual(len(commands),8)
            self.assertTrue(result['protected_v6_verified_before']);self.assertTrue(result['protected_v6_verified_after'])
            self.assertTrue(all(e['CUBLAS_WORKSPACE_CONFIG']==':4096:8' for e in environments))
            self.assertEqual(sum(e['CUDA_VISIBLE_DEVICES']=='4' for e in environments),3)
            self.assertTrue(all(e['OMP_NUM_THREADS']=='2' for e in environments))
            self.assertFalse(result['local_callback_required']);self.assertFalse(result['scheduled_monitor'])

    def test_nonfourth_gpu_is_rejected_before_any_probe(self):
        with self.assertRaisesRegex(RuntimeError,'only physical GPU4'):p.gpu_check(6,1)

if __name__=='__main__':unittest.main()
