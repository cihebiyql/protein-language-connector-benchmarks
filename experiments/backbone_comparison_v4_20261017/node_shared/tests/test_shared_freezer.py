"""Targeted synthetic tests for the explicitly shared node1 freezer only."""
from datetime import datetime,timedelta,timezone
import importlib.util,json
from pathlib import Path
import tempfile
import unittest

SCOPE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared_freezer_fixture',SCOPE/'resources/freeze_formal_shared.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class SharedFreezerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.scope=Path(self.temp.name)
        (self.scope/'resources').mkdir();self.policy=self.scope/'USER_RESOURCE_POLICY.json'
        self.policy.write_text(json.dumps({'id':'user_shared_node1_v1','fixture_only':True}),encoding='utf8')
        self.now=datetime.now(timezone.utc)
        self.obs={'utc':self.now.isoformat(),'hostname':'node1','gpu':4,'readiness':'ready','reasons':[],
            'user_resource_policy_id':'user_shared_node1_v1','user_resource_policy_sha256':f.sha(self.policy),
            'gpu_memory_budget_bytes':20*f.GIB,'gpu_free_headroom_bytes':512*1024**2,'host_ram_reserve_bytes':4*f.GIB,
            'memory_available_bytes':100*f.GIB,'cpu_idle_percent':0.,'cpu_iowait_percent':90.,
            'gpu_frame':{'free_mib':22000,'utilization':99,'used_mib':2000},'gpu_compute_processes':['fixture_foreign_existing_compute']}
        self.path=self.scope/'resources/ready4.json';self.save()
    def save(self):self.path.write_text(json.dumps(self.obs),encoding='utf8')
    def tearDown(self):self.temp.cleanup()
    def accept(self):return f.ready_assignment(4,self.path,'node1',self.scope,self.now)
    def test_CPU_util_and_foreign_compute_do_not_block_user_authorized_sharing(self):
        self.assertEqual(self.accept()['gpu'],4)
        self.assertEqual(f.SELECTED_GPUS,(4,5,3))
    def test_actual_RAM_and_VRAM_headroom_remain_mandatory(self):
        self.obs['memory_available_bytes']=67*f.GIB;self.save()
        with self.assertRaises(ValueError):self.accept()
        self.obs['memory_available_bytes']=100*f.GIB;self.obs['gpu_frame']['free_mib']=20991;self.save()
        with self.assertRaises(ValueError):self.accept()
    def test_stale_or_wrong_policy_byte_binding_rejected(self):
        self.obs['utc']=(self.now-timedelta(minutes=4)).isoformat();self.save()
        with self.assertRaises(ValueError):self.accept()
        self.obs['utc']=self.now.isoformat();self.obs['user_resource_policy_sha256']='0'*64;self.save()
        with self.assertRaises(ValueError):self.accept()
    def test_unselected_device_rejected(self):
        self.obs['gpu']=7;self.save()
        with self.assertRaises(ValueError):f.ready_assignment(7,self.path,'node1',self.scope,self.now)


if __name__=='__main__':unittest.main()
