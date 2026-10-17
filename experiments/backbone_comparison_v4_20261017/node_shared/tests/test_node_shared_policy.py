"""User-authorized contention sharing; actual memory and scope limits remain gated."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from src import mm_supervise as s
from src import mm_queue as q

GIB=1024**3


class SharedPolicyTests(unittest.TestCase):
    def probe(self,*,available=400*GIB,free=21000,util='99',processes='GPU-fixture, 123, 4096',host='node1',writable=True,storage=10*GIB):
        def proc(path,*args,**kwargs):
            if str(path)=='/proc/meminfo':return f'MemAvailable: {available//1024} kB\n'
            return 'some avg10=99.00 avg60=99.00 avg300=99.00 total=100\n'
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(s.socket,'gethostname',return_value=host), \
             patch.object(s,'cpu_frame',side_effect=[(1000,0,0),(2000,10,900)]), \
             patch.object(s.time,'sleep'),patch.object(s.os,'getloadavg',return_value=(1000,1000,1000)), \
             patch.object(s.Path,'read_text',proc), \
             patch.object(s.os,'access',return_value=writable), \
             patch.object(s.os,'statvfs',return_value=SimpleNamespace(f_bavail=storage,f_frsize=1)), \
             patch.object(s.subprocess,'check_output',side_effect=[f'GPU-fixture, 8000, {free}, {util}',processes]):
            return s.probe('node1',4,64*GIB,Path(directory))

    def test_busy_cpu_io_utilization_and_other_processes_are_observed_and_allowed(self):
        value=self.probe()
        self.assertEqual(value['readiness'],'ready')
        self.assertEqual(value['reasons'],[])
        self.assertLess(value['cpu_idle_percent'],8)
        self.assertGreater(value['cpu_iowait_percent'],10)
        self.assertEqual(value['gpu_frame']['utilization'],99)
        self.assertTrue(value['gpu_compute_processes'])
        self.assertTrue(value['contention_is_observation_only'])
        self.assertEqual(value['gpu_memory_budget_bytes'],20*GIB)

    def test_utilization_unknown_or_nonempty_unknown_process_rows_are_not_capacity_gates(self):
        value=self.probe(util='N/A',processes='N/A, 123, 100')
        self.assertEqual(value['readiness'],'ready')
        self.assertIsNone(value['gpu_frame']['utilization'])
        self.assertEqual(value['all_gpu_compute_process_rows_observed'],['N/A, 123, 100'])

    def test_actual_ram_and_vram_boundary_remain_closed(self):
        self.assertEqual(self.probe(available=68*GIB,free=20992)['readiness'],'ready')
        self.assertIn('insufficient_host_memory',self.probe(available=68*GIB-1024)['reasons'])
        self.assertIn('insufficient_gpu_free_memory',self.probe(free=20991)['reasons'])

    def test_host_output_permission_and_storage_remain_required(self):
        self.assertIn('hostname_mismatch',self.probe(host='foreign')['reasons'])
        self.assertIn('output_path_unwritable',self.probe(writable=False)['reasons'])
        self.assertIn('insufficient_storage',self.probe(storage=10*GIB-1)['reasons'])

    def test_budget_cannot_silently_increase(self):
        for value in (0,True,21*GIB):
            with self.assertRaises(ValueError):s.probe('node1',4,64*GIB,Path('/tmp'),gpu_memory_budget_bytes=value)

    def test_queue_permits_sharing_but_rejects_four_candidates_or_over_budget(self):
        from tests.test_queue import fixture
        with tempfile.TemporaryDirectory() as directory:
            _,plan,_=fixture(Path(directory),gpus=(4,5,3))
            plan['occupied_gpu_devices']=[3,4,5]
            plan['candidate_gpu_devices']=[4,5,3]
            plan['contract_request']['resources']['memory_limit_bytes']=64*GIB
            self.assertTrue(q.validate_plan(plan))
            for change in ('four','cpu','ram','vram'):
                bad=deepcopy(plan)
                if change=='four':bad['candidate_gpu_devices']=[4,5,3,2]
                if change=='cpu':bad['contract_request']['resources']['cpu_threads']=3
                if change=='ram':bad['contract_request']['resources']['memory_limit_bytes']=64*GIB+1
                if change=='vram':bad['gpu_memory_budget_bytes']=21*GIB;bad['contract_request']['resources']['gpu_memory_budget_bytes']=21*GIB
                with self.subTest(change=change),self.assertRaises(ValueError):q.validate_plan(bad)
