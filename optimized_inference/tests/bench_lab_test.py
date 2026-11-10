"""Behavioral contracts for root control logic; no privileged writes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('lab', Path(__file__).resolve().parents[1] / 'tools/bench_lab/controller.py')
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class LabContracts(unittest.TestCase):
    def test_cpu_masks_across_32bit_boundary(self):
        self.assertEqual(lab.cpus('0-3,12-15'), {0,1,2,3,12,13,14,15})
        self.assertEqual(lab.cpumask({0,32,63}), '80000001,00000001')
        with self.assertRaises(ValueError):
            lab.cpus('1;reboot')

    def test_discovers_cache_domain_not_fixed_cpu_numbers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'online').write_text('0-7')
            for path, value in {'cpu3/cache/index3/level':'3', 'cpu3/cache/index3/type':'Unified',
                                'cpu3/cache/index3/shared_cpu_list':'2-3,6-7',
                                'cpu3/topology/thread_siblings_list':'3,7'}.items():
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
            with patch.object(lab,'CPU',root):
                online,reserved,siblings,house=lab.topology(3)
            self.assertEqual(reserved,{2,3,6,7})
            self.assertEqual(siblings,{3,7})
            self.assertEqual(house,{0,1,4,5})

    def test_journal_is_durable_before_write(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'setting';p.write_text('old')
            state={'journal':[]};saved=[]
            original=lab.write
            def observed(path,value):
                self.assertTrue(saved)
                self.assertEqual(saved[-1][0]['before'],'old')
                original(path,value)
            with patch.object(lab,'save',lambda s:saved.append([x.copy() for x in s['journal']])),patch.object(lab,'write',observed):
                lab.change(state,p,'new')
            self.assertEqual(state['journal'][0]['applied'],'new')

    def test_optional_unwritable_irq_is_retained_as_diagnostic(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'irq';p.write_text('0-3');state={'journal':[]}
            with patch.object(lab,'save'),patch.object(lab,'write',side_effect=PermissionError('managed IRQ')):
                lab.change(state,p,'4-7',optional=True)
            self.assertIn('error',state['journal'][0])
            self.assertEqual(p.read_text(),'0-3')

    def test_frequency_bound_order_handles_both_directions(self):
        for initial,target in [((100,200),(300,300)),((300,400),(100,100)),((100,400),(200,200))]:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)
                for name,value in zip(['scaling_min_freq','scaling_max_freq'],initial):(p/name).write_text(str(value))
                original=lab.write
                def checked(path,value):
                    low=int(lab.read(p/'scaling_min_freq'));high=int(lab.read(p/'scaling_max_freq'))
                    self.assertTrue(int(value)<=high if Path(path).name=='scaling_min_freq' else int(value)>=low)
                    original(path,value)
                with patch.object(lab,'write',checked):lab.set_limits(p,*target)
                self.assertEqual((int(lab.read(p/'scaling_min_freq')),int(lab.read(p/'scaling_max_freq'))),target)

    def test_zombie_is_not_a_runnable_foreign_thread(self):
        with patch.object(Path,'glob',return_value=[Path('/proc/42')]),patch.object(lab,'identity',return_value=dict(state='Z',flags=0)),patch.object(lab.os,'sched_getaffinity',side_effect=AssertionError('zombie must not count')):
            self.assertEqual(lab.foreign_threads({'reserved':[0,1]}),[])

    def test_changed_probe_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'probe';p.write_text('changed')
            with patch.object(lab,'verify'),patch.object(lab.subprocess,'check_output',side_effect=AssertionError('must not execute')):
                with self.assertRaises(RuntimeError):lab.calibrate({'probe':str(p),'probe_sha256':'old'})

    def test_boost_validation_is_not_fixed_frequency_validation(self):
        r=[{'effective_khz':v} for v in [2000000,4000000,5000000]]
        s=dict(power_profile='boost',worker_power=dict(scaling_min_freq='600000',scaling_max_freq='5300000'))
        self.assertTrue(lab.frequency_qualified(r,s))
        self.assertFalse(lab.frequency_qualified(r,dict(power_profile='steady',frequency_khz=2000000)))
        self.assertFalse(lab.frequency_qualified([{'effective_khz':9000000}],s))

    def test_completed_restore_is_idempotent(self):
        result={'restored':True,'errors':[]}
        with patch.object(lab,'write',side_effect=AssertionError('unexpected mutation')):
            self.assertEqual(lab.restore({'phase':'restored','restoration':result}),result)


class StudyContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('study',Path(__file__).resolve().parents[1]/'tools/bench_lab/study.py')
        cls.study=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.study)

    def test_controller_failure_preserves_reason_and_rollback_verdict(self):
        detail='{"error":"Frequency policy changed: scaling_min_freq","phase":"restored","restoration":{"restored":true}}'
        failure=self.study.subprocess.CalledProcessError(1,['controller','prepare'],output=detail)
        with patch.object(self.study.subprocess,'check_output',side_effect=failure):
            with self.assertRaises(RuntimeError) as captured:
                self.study.control('prepare')
        self.assertIn('Frequency policy changed: scaling_min_freq',str(captured.exception))
        self.assertIn('"restored":true',str(captured.exception))

    def test_order_balances_within_each_input(self):
        from collections import defaultdict,Counter
        m=dict(seed=123,aa_blocks=4,compare_blocks=8,variants={'native':{},'avx2':{}},models=[2,8],inputs=['x','y'])
        for kind in ['aa','compare']:
            jobs=self.study.schedule(m,kind);orders=defaultdict(Counter)
            for j in jobs[::2]:orders[j['isa'],j['model'],j['mode'],j['input']][j['label']]+=1
            for v in orders.values():self.assertEqual(v['A'],v['B'])
            if kind=='aa':self.assertTrue(all(j['arm']=='baseline' for j in jobs))

    def test_custom_condition_order_and_cadence(self):
        m=dict(seed=9,aa_blocks=8,compare_blocks=8,models=[2,8],variants={'native':{'baseline':'/a','candidate':'/b'}},inputs=['x','y'],worker='/worker',weights={'2':'/w2','8':'/w8'},
               conditions=[dict(id='batch64',worker_mode='batch',batch_size=64,calls=512),dict(id='paced20',worker_mode='paced',batch_size=1,calls=64,period_ns=20000000)])
        jobs=self.study.schedule(m,'aa')
        self.assertEqual(len(jobs),64)
        j=next(j for j in jobs if j['mode']=='paced20')
        args=self.study.worker_args(m,j,Path('/raw'))
        self.assertEqual(args[-3:],["paced:20000000","1","/raw"])
        self.assertEqual(len(set((j['mode'],j['model']) for j in jobs)),4)

    def test_summary_handles_explicit_family_and_conditions(self):
        import array
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[]
            for block in range(4):
                for label in ['A','B']:
                    name=f'{block}-{label}.f64';(root/name).write_bytes(array.array('d',[1.,1.]).tobytes())
                    rows.append(dict(block=block,isa='native',model=2,mode='batch16',label=label,samples=name,
                        result=dict(mean_ms=1.,calls=2,observation_unit='batch_mean',reference_delta=2,delivered_delta=2,late=0,over_10ms=0,minor_faults=0,major_faults=0)))
            m=dict(models=[2],variants={'native':{}},family_comparisons=48,conditions=[dict(id='batch16')])
            result=self.study.summarize(m,rows,'aa',root)
            self.assertEqual(result['family_comparisons'],48)
            self.assertEqual(result['conditions'][0]['gain_percent'],0.)
            self.assertTrue(result['conditions'][0]['sensitivity_qualified'])

    def test_callback_uses_registered_worker_and_pcm_inputs(self):
        m=dict(seed=3,aa_blocks=4,compare_blocks=4,models=[2],variants={'native':{'baseline':'/a','candidate':'/b'}},inputs=['spectra-x','spectra-y'],worker='/nn',weights={'2':'/w2'},
               conditions=[dict(id='audio10',worker_mode='paced',worker='/audio',inputs=['pcm-x','pcm-y'],period_ns=10000000,batch_size=1,calls=32)])
        jobs=self.study.schedule(m,'aa');self.assertEqual(set(j['input'] for j in jobs),{'pcm-x','pcm-y'})
        self.assertEqual(self.study.worker_args(m,jobs[0],Path('/raw'))[0],'/audio')

    def test_no_effect_not_a_speedup(self):
        r=self.study.effect([0.]*4,.05/8)
        self.assertEqual(r['gain_percent'],0)
        self.assertEqual(r['interval_percent'],[0,0])

    def test_retains_slow_observations_in_sd_and_tail(self):
        r=self.study.describe([1.]*99+[20.])
        self.assertEqual(r['n'],100);self.assertEqual(r['p99'],20.)
        self.assertGreater(r['sd'],1.)

    def test_comparison_requires_identical_binary_control(self):
        with patch.object(self.study,'binary_manifest',return_value={'binary':'new'}):
            with self.assertRaises(ValueError):
                self.study.timing({},'compare','never-created',dict(kind='aa',hashes={'binary':'old'}))


if __name__=='__main__':
    unittest.main()
